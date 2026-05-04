import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import websockets

import sys
sys.path.insert(0, str(Path(__file__).parent))

from bcg_core.config_schema import AppConfig
from bcg_core.cortex_reader import CortexReader
from bcg_core.classifier import EEGPreprocessor, RealtimeClassifier

logger = logging.getLogger(__name__)


class EEGSource:
    def __init__(self, app_cfg: AppConfig):
        self._app_cfg    = app_cfg
        self._n_channels = app_cfg.model.n_channels
        self.on_sample   = None
        self._reader: Optional[CortexReader] = None

    def start(self):
        self._reader = CortexReader(self._app_cfg)
        self._reader.on_sample = self._on_real_sample
        self._reader.start()
        logger.info("CortexReader started — waiting for headset...")

    def stop(self):
        if self._reader:
            self._reader.stop()

    def _on_real_sample(self, sample: np.ndarray):
        if self.on_sample:
            self.on_sample(sample.tolist())


def run_calibration(trials: list, cfg: dict) -> str:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    try:
        from braindecode.models import EEGNetv4 as EEGNet
    except ImportError:
        from braindecode.models import EEGNet

    n_ch    = cfg["model"]["n_channels"]
    n_out   = cfg["live"]["n_outputs"]
    epochs  = cfg["server"]["calibration_epochs"]
    classes = cfg["live"]["classes"]
    sr      = cfg["preprocessing"]["sampling_rate"]

    model_path  = Path(__file__).parent / cfg["live"]["model_path"]
    output_path = Path(__file__).parent / "model" / "models" / "eegnet_game_calibrated.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Calibrating on {device} | {len(trials)} trials | {epochs} epochs")

    preprocessor = EEGPreprocessor(sampling_rate=sr)
    label_map    = {cls: i for i, cls in enumerate(classes)}
    X_list, y_list = [], []

    for t in trials:
        label = t.get("label", "")
        if label not in label_map:
            continue
        eeg = np.array(t["eeg"], dtype=np.float32)
        if eeg.ndim != 2 or eeg.shape[0] != n_ch or eeg.shape[1] < 64:
            logger.warning(f"Skipping trial — bad shape {eeg.shape}")
            continue
        X_list.append(preprocessor.process(eeg))
        y_list.append(label_map[label])

    if len(X_list) < 4:
        raise ValueError(f"Only {len(X_list)} valid trials — need at least 4.")

    n_times = X_list[0].shape[1]
    X       = torch.tensor(np.stack(X_list)[:, np.newaxis], dtype=torch.float32)
    y       = torch.tensor(y_list, dtype=torch.long)
    loader  = DataLoader(TensorDataset(X, y), batch_size=min(16, len(X_list)), shuffle=True)

    net = EEGNet(n_chans=n_ch, n_outputs=n_out, n_times=n_times, drop_prob=0.5).to(device)

    if model_path.exists():
        raw   = torch.load(model_path, map_location=device)
        state = raw.get("model_state_dict") or raw.get("state_dict") or raw
        state = {k.replace(".parametrizations.weight.original", ".weight"): v for k, v in state.items()}
        net.load_state_dict(state, strict=False)
        logger.info(f"✓ Loaded pretrained weights ← {model_path}")
    else:
        logger.warning(f"Pretrained model not found — using random weights")

    opt     = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    net.train()

    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        sch.step()
        if (ep + 1) % 50 == 0:
            logger.info(f"  epoch {ep+1}/{epochs}  loss={loss.item():.4f}")

    torch.save({"model_state_dict": net.state_dict()}, output_path)
    logger.info(f"✓ Calibrated model saved → {output_path}")
    return str(output_path)


class BCGServer:
    def __init__(self, cfg: dict, signals=None):
        self._cfg     = cfg
        self._srv     = cfg["server"]
        self._signals = signals   # ServerSignals | None

        app_cfg    = AppConfig.model_validate(cfg)
        self._eeg  = EEGSource(app_cfg)

        self._clients: set = set()
        self._phase        = "collection"
        self._trials: list = []
        self._clf: Optional[RealtimeClassifier] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        sr             = cfg["preprocessing"]["sampling_rate"]
        self._n_times  = int(sr * self._srv["trial_seconds"])
        self._step     = cfg["live"]["step_samples"]
        self._buf: list = []
        self._step_ctr  = 0

    def _emit(self, signal_name: str, *args):
        if self._signals:
            getattr(self._signals, signal_name).emit(*args)

    def _on_sample(self, sample: list):
        self._buf.append(sample)
        if len(self._buf) > self._n_times * 4:
            self._buf = self._buf[-(self._n_times * 4):]

        if self._phase == "inference" and self._clf:
            if len(self._buf) < self._n_times:
                return
            self._step_ctr += 1
            if self._step_ctr >= self._step:
                self._step_ctr = 0
                eeg = np.array(self._buf[-self._n_times:], dtype=np.float32).T
                label, conf = self._clf.predict(eeg)
                pred = json.dumps({"type": "prediction", "label": label, "confidence": round(conf, 1)})
                asyncio.run_coroutine_threadsafe(self._broadcast(pred), self._loop)

    async def _broadcast(self, msg: str):
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def _handler(self, websocket):
        self._clients.add(websocket)
        self._emit("log", f"+ Client connected: {websocket.remote_address}")
        self._emit("clients", len(self._clients))
        await websocket.send(json.dumps({"type": "status", "phase": self._phase}))
        try:
            async for raw in websocket:
                await self._on_message(websocket, raw)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self._clients.discard(websocket)
            self._emit("log", f"- Client disconnected")
            self._emit("clients", len(self._clients))

    async def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
            return

        t = msg.get("type")

        if t == "trial_marker":
            label  = msg["label"]
            window = self._buf[-self._n_times:] if len(self._buf) >= self._n_times else self._buf[:]
            while len(window) < self._n_times:
                window = [window[0]] + window
            self._trials.append({"label": label, "eeg": np.array(window, dtype=np.float32).T.tolist()})
            self._emit("log", f"Trial {len(self._trials)}: {label}")
            self._emit("trial_count", len(self._trials))

        elif t == "start_calibration":
            trials = self._trials[:]
            self._emit("log", f"Calibrating on {len(trials)} trials...")
            self._emit("phase", "calibrating")
            self._phase = "calibrating"
            try:
                model_path = await asyncio.get_event_loop().run_in_executor(
                    None, run_calibration, trials, self._cfg
                )
                self._clf = RealtimeClassifier(
                    checkpoint_path      = model_path,
                    n_channels           = self._cfg["model"]["n_channels"],
                    n_outputs            = self._cfg["live"]["n_outputs"],
                    n_times              = self._n_times,
                    confidence_threshold = self._cfg["live"]["confidence_threshold"],
                    preprocessing_config = {
                        "sampling_rate": self._cfg["preprocessing"]["sampling_rate"],
                        "l_freq": 8, "h_freq": 30, "notch_freq": 50
                    }
                )
                self._phase = "inference"
                await self._broadcast(json.dumps({"type": "calibration_done"}))
                self._emit("phase", "inference")
                self._emit("log", "✓ Calibration done — live inference started")
            except Exception as e:
                logger.exception("Calibration failed")
                self._phase = "collection"
                self._emit("phase", "collection")
                self._emit("log", f"✗ Calibration failed: {e}")
                await self._broadcast(json.dumps({"type": "error", "message": str(e)}))

        elif t == "ping":
            await ws.send(json.dumps({"type": "pong"}))

    async def run(self):
        self._loop = asyncio.get_event_loop()
        self._eeg.on_sample = self._on_sample
        self._eeg.start()
        self._emit("headset", True)

        host = self._srv["host"]
        port = self._srv["port"]
        self._emit("log", f"✓ Server running on ws://{host}:{port}")
        self._emit("phase", "collection")

        async with websockets.serve(self._handler, host, port):
            await asyncio.Future()