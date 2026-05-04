import asyncio
import json
import logging
import threading
import time
import websockets
import torch

import torch.nn as nn
import numpy as np
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader
from braindecode.models import EEGNet
from bcg_core.classifier import EEGPreprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "server_conf.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---> 
class BCGServer:
    def __init__(self, cfg: dict):
        self._cfg = cfg

        # Network
        self._host = cfg["server"]["host"]
        self._port = cfg["server"]["port"]

        # EEG
        self._sampling_rate = cfg["preprocessing"]["sampling_rate"]
        self._n_channels = cfg["model"]["n_channels"]
        self._step_samples = cfg["live"]["step_samples"]
        self._simulation = cfg["live"]["simulation_mode"]
        self._trials = []
        self._trial_samples = int(
            cfg["preprocessing"]["sampling_rate"] * cfg["server"]["trial_seconds"]
        )

        # State
        self._client = None
        self._phase = "collection"
        self._buf = []
        self._stream_ctr = 0
        self._loop = None

        # Model
        self._clf = None

        
    def _on_sample(self, sample: list):
        self._buf.append(sample)

        self._stream_ctr += 1
        if self._stream_ctr >= self._step_samples:
            self._stream_ctr = 0
            window = self._buf[-self._step_samples:]
            asyncio.run_coroutine_threadsafe(
                self.send({"type": "eeg_window", "data": window}),
                self._loop
            )


    async def send(self, msg: dict):
        if self._client:
            await self._client.send(json.dumps(msg))


    async def _handler(self, websocket):
        if self._client is not None:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Server already has a client"
            }))
            await websocket.close()
            return

        self._client = websocket
        logger.info(f"+ Client: {websocket.remote_address}")
        await self.send({"type": "status", "phase": self._phase})

        try:
            async for raw in websocket:
                await self._on_message(raw)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        finally:
            self._client = None
            logger.info("- Client disconnected")


    async def _on_message(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self.send({"type": "error", "message": "Invalid JSON"})
            return

        t = msg.get("type")

        if t == "ping":
            await self.send({"type": "pong"})
        elif t == "trial_marker":
            await self._handle_trial_marker(msg)
        elif t == "start_calibration":
            await self._handle_start_calibration()
        elif t == "start_inference":
            await self._handle_start_inference()
        else:
            logger.warning(f"Unknown message type: {t}")


    async def _handle_trial_marker(self, msg: dict):
        if self._phase != "collection":
            return

        label = msg.get("label")
        if label not in ("Left Hand", "Right Hand"):
            await self.send({"type": "error", "message": f"Invalid label: {label}"})
            return

        trial_samples = int(self._sampling_rate * self._cfg["server"]["trial_seconds"])

        if len(self._buf) < trial_samples:
            await self.send({"type": "error", "message": "Buffer not full yet"})
            return

        eeg = self._buf[-trial_samples:]  
        self._trials.append({"label": label, "eeg": eeg})

        n = len(self._trials)
        logger.info(f"Trial {n}: {label}")
        await self.send({"type": "trial_count", "count": n})


    async def _handle_start_inference(self):
        if self._clf is None:
            await self.send({"type": "error", "message": "No model loaded"})
            return

        self._phase = "inference"
        await self.send({"type": "phase_change", "phase": "inference"})
        logger.info("Phase → inference")

    
    async def _handle_start_calibration(self):
        if self._phase != "collection":
            return
        if len(self._trials) < 6:  
            await self.send({"type": "error", "message": "Not enough trials"})
            return

        self._phase = "calibrating"
        await self.send({"type": "phase_change", "phase": "calibrating"})

        loop = asyncio.get_event_loop()
        try:
            model_path = await loop.run_in_executor(None, self._run_calibration)
            self._load_classifier(model_path)
            self._phase = "inference"
            await self.send({"type": "calibration_done"})
            await self.send({"type": "phase_change", "phase": "inference"})
        except Exception as e:
            logger.exception("Calibration failed")
            self._phase = "collection"
            await self.send({"type": "phase_change", "phase": "collection"})
            await self.send({"type": "error", "message": str(e)})

    def _run_calibration(self) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sr = self._cfg["preprocessing"]["sampling_rate"]
        n_ch = self._cfg["model"]["n_channels"]
        epochs = self._cfg["server"]["calibration_epochs"]
        n_times = int(sr * self._cfg["server"]["trial_seconds"])

        pretrained_path = Path(__file__).parent / self._cfg["live"]["model_path"]
        output_path = Path(__file__).parent / "models" / "eegnet_game_calibrated.pth"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---> Prepare data
        preprocessor = EEGPreprocessor(sampling_rate=sr)
        label_map = {"Left Hand": 0, "Right Hand": 1, "Rest": 2}

        X_list, y_list = [], []
        for trial in self._trials:
            label = trial["label"]
            if label not in label_map:
                continue
            eeg = np.array(trial["eeg"], dtype=np.float32)  # (n_ch, n_times)
            if eeg.shape != (n_ch, n_times):
                logger.warning(f"Skip trial — bad shape {eeg.shape}")
                continue
            X_list.append(preprocessor.process(eeg))
            y_list.append(label_map[label])

        if len(X_list) < 6:
            raise ValueError(f"Only {len(X_list)} valid trials — need at least 6")

        X = torch.tensor(np.stack(X_list)[:, np.newaxis], dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.long)

        # Imbalance handle
        counts = np.bincount(y_list, minlength=3).astype(np.float32)
        counts = np.where(counts == 0, 1, counts)
        weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)

        loader = DataLoader(TensorDataset(X, y), batch_size=min(16, len(X_list)), shuffle=True)
    # <---

    # ---> Build model
        model = EEGNet(n_chans=n_ch, n_outputs=3, n_times=n_times, drop_prob=0.5).to(device)

        # Load pretrained weights 
        if pretrained_path.exists():
            ckpt = torch.load(pretrained_path, map_location=device)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            state = {k.replace(".parametrizations.weight.original", ".weight"): v
                     for k, v in state.items()}
            model_state = model.state_dict()
            transferred = {k: v for k, v in state.items()
                           if k in model_state and model_state[k].shape == v.shape}
            model_state.update(transferred)
            model.load_state_dict(model_state)
            logger.info(f"Transferred {len(transferred)} layers from pretrained")
        else:
            logger.warning("Pretrained model not found — using random weights")

        # Freeze temporal conv, train spatial + classifier
        for name, param in model.named_parameters():
            if "conv_temporal" in name or "bnorm_temporal" in name:
                param.requires_grad = False

    # <---

        # ── Train ─────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        model.train()
        for ep in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss_fn(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()
            if (ep + 1) % 50 == 0:
                logger.info(f"Epoch {ep+1}/{epochs}")

        torch.save({"model_state_dict": model.state_dict()}, output_path)
        logger.info(f"✓ Model saved → {output_path}")
        return str(output_path)
# <---