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
from datetime import datetime

from torch.utils.data import TensorDataset, DataLoader
try:
    from braindecode.models import EEGNetv4 as EEGNet
except ImportError:
    from braindecode.models import EEGNet
from bcg_core.classifier import EEGPreprocessor, RealtimeClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "server_conf.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---> 
class BCGServer:
    def __init__(self, cfg: dict, signals=None):
        self._cfg = cfg
        self._signals = signals

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
        self._sample_counter = 0

        # State
        self._client = None
        self._phase = "collection"
        self._buf = []
        self._stream_ctr = 0
        self._loop = None
        self._trials = []
        self._current_trial = None

        # Track current trial
        self._current_trial = None

        # Model
        self._clf = None

        self._inference_wait_seconds = 0.0
        
        # Simulation thread
        self._running = False
        self._eeg_thread = None

        # Saving
        self._save_dir = Path(cfg.get("server", {}).get("save_dir") or (Path(__file__).parent.parent / "recordings"))
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._min_trials_to_save = cfg.get("server", {}).get("min_trials_to_save")
        try:
            self._min_trials_to_save = int(self._min_trials_to_save) if self._min_trials_to_save is not None else None
        except Exception:
            self._min_trials_to_save = None
        self._ready_to_save_sent = False
        
    def _on_sample(self, sample: list):
        self._sample_counter += 1   
        if len(self._buf) % 50 == 0:
            logger.info(f"Buffer size: {len(self._buf)}")
        self._buf.append(sample)
        log_eeg = f"{len(self._buf)} | " + " ".join(f"{v:.1f}" for v in sample[:4])
        self._emit("eeg", log_eeg)

        # Prevent unbounded growth
        if len(self._buf) > self._trial_samples * 4:
            self._buf = self._buf[-self._trial_samples * 4:]

        self._stream_ctr += 1

        # Inference phase: send predictions
        if self._phase == "inference" and self._clf:
            if len(self._buf) < self._trial_samples:
                return
            if self._stream_ctr < self._step_samples:
                return

            self._stream_ctr = 0
            window = self._buf[-self._trial_samples:]
            eeg = np.array(window, dtype=np.float32).T  

            try:
                label, conf = self._clf.predict(eeg)
                asyncio.run_coroutine_threadsafe(
                    self.send({
                        "type": "prediction",
                        "label": label,
                        "confidence": round(conf, 2),
                    }),
                    self._loop,
                )
            except Exception as e:
                logger.error(f"Prediction error: {e}")
            return

        # Collection phase: keep streaming eeg_window for debugging/game
        if self._phase == "collection":
            if self._stream_ctr >= self._step_samples:
                self._stream_ctr = 0
                window = self._buf[-self._step_samples:]
                asyncio.run_coroutine_threadsafe(
                    self.send({"type": "eeg_window", "data": window}),
                    self._loop,
                )

    def _emit(self, name: str, *args):
        if self._signals is None:
            return
        sig = getattr(self._signals, name, None)
        if sig is not None:
            sig.emit(*args)

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
        self._emit("log", f"+ Client: {websocket.remote_address}")
        self._emit("clients", 1)
        await self.send({"type": "status", "phase": self._phase})

        try:
            async for raw in websocket:
                await self._on_message(raw)
        except websockets.exceptions.ConnectionClosedOK:
            pass
        finally:
            self._client = None
            logger.info("- Client disconnected")
            self._emit("log", "- Client disconnected")
            self._emit("clients", 0)


    async def _on_message(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self.send({"type": "error", "message": "Invalid JSON"})
            return

        t = msg.get("type")

        if t == "ping":
            await self.send({"type": "pong"})
        elif t == "trial_start":
            await self._handle_trial_start(msg)
        elif t == "trial_end":
            await self._handle_trial_end(msg)
        elif t == "save_game_session":
            await self._handle_save_game_session(msg)
        elif t == "start_calibration":
            await self._handle_start_calibration()
        elif t == "start_inference":
            await self._handle_start_inference()
        else:
            logger.warning(f"Unknown message type: {t}")

    async def run(self):
        self._loop = asyncio.get_event_loop()

        logger.info(f"✓ Server running on ws://{self._host}:{self._port}")
        self._emit("log", f"✓ Server running on ws://{self._host}:{self._port}")
        self._emit("phase", "collection")

        async with websockets.serve(self._handler, self._host, self._port):
            await asyncio.Future()

    async def _handle_trial_start(self, msg: dict):
        if self._phase != "collection":
            return

        label = msg.get("label")
        if label not in ("Left Hand", "Right Hand", "Rest"):
            await self.send({"type": "error", "message": f"Invalid label: {label}"})
            return

        trial_samples = int(self._sampling_rate * self._cfg["server"]["trial_seconds"])

        self._current_trial = {
            "label": label,
            "start_counter": self._sample_counter
        }
        logger.info(f"Trial start: {label} at counter {self._sample_counter}")


    async def _handle_trial_end(self, msg: dict):
        if self._phase != "collection":
            return
        if not self._current_trial:
            # End without start -> ignore
            return

        # Optionally check label consistency
        end_label = msg.get("label", self._current_trial["label"])
        if end_label != self._current_trial["label"]:
            logger.warning(f"Label mismatch: start={self._current_trial['label']} end={end_label}")

        start_counter = self._current_trial["start_counter"]  
        n_samples = self._sample_counter - start_counter       

        buf_len = len(self._buf)
        samples_in_buf = min(n_samples, buf_len)
        window = self._buf[buf_len - samples_in_buf:]

        if n_samples < int(self._trial_samples * 0.8):
            msg = (
                f"Trial window too short "
                f"(label={self._current_trial['label']}, "
                f"samples={n_samples}, "
                f"expected>={int(self._trial_samples * 0.8)})"
            )

            logger.warning(msg)
            await self.send({"type": "error", "message": msg})
            self._current_trial = None
            return

        # Trim / pad to exact length
        target = self._trial_samples
        if n_samples > target:
            window = window[-target:]
        elif n_samples < target:
            while len(window) < target:
                window.append(window[-1])

        eeg = np.array(window, dtype=np.float32).T
        if eeg.shape[0] != self._n_channels:
            logger.warning(f"Trial bad shape after slice: {eeg.shape}")
            self._current_trial = None
            return

        self._trials.append({"label": self._current_trial["label"], "eeg": eeg})
        n = len(self._trials)
        logger.info(f"Trial {n}: {self._current_trial['label']} — shape={eeg.shape}")

        # Notify client about count
        await self.send({"type": "trial_count", "count": n})
        self._emit("trial_count", n)

        if (
            (self._min_trials_to_save is not None)
            and (not self._ready_to_save_sent)
            and (n >= self._min_trials_to_save)
        ):
            self._ready_to_save_sent = True
            await self.send({"type": "ready_to_save", "count": n, "min_trials": self._min_trials_to_save})

        self._current_trial = None

    async def _handle_save_game_session(self, msg: dict):
        if self._phase != "collection":
            await self.send({"type": "error", "message": "Can only save during collection phase"})
            return

        min_trials = msg.get("min_trials")
        if isinstance(min_trials, (int, float)) and int(min_trials) > 0:
            min_trials = int(min_trials)
            if len(self._trials) < min_trials:
                await self.send({
                    "type": "error",
                    "message": f"Not enough trials to save: have {len(self._trials)} need {min_trials}",
                })
                return

        try:
            path = self._save_game_trials_npz()
        except Exception as e:
            logger.exception("Save failed")
            await self.send({"type": "error", "message": f"Save failed: {e}"})
            return

        await self.send({"type": "session_saved", "method": "game", "path": str(path), "trials": len(self._trials)})
        self._emit("log", f"✓ Saved game session → {path.name}")

    def _save_game_trials_npz(self) -> Path:
        # Trials store eeg as (n_ch, n_times). Save as (n_trials, n_times, n_ch) to match bcg_collect.
        if not self._trials:
            raise ValueError("No trials to save")

        classes = self._cfg.get("live", {}).get("classes") or ["Left Hand", "Rest", "Right Hand"]
        label_map = {cls: i for i, cls in enumerate(classes)}

        eeg_list = []
        y_list = []
        for t in self._trials:
            label = t.get("label")
            if label not in label_map:
                continue
            eeg = t.get("eeg")
            if not isinstance(eeg, np.ndarray):
                eeg = np.asarray(eeg, dtype=np.float32)
            if eeg.ndim != 2:
                continue
            # (n_ch, n_times) -> (n_times, n_ch)
            eeg_list.append(eeg.T.astype(np.float32))
            y_list.append(label_map[label])

        if not eeg_list:
            raise ValueError("No valid trials (label mismatch or bad shapes)")

        eeg_arr = np.stack(eeg_list, axis=0).astype(np.float32)
        labels_arr = np.asarray(y_list, dtype=np.int32)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"bcg_game_session_{ts}.npz"
        out_path = self._save_dir / file_name

        np.savez(
            out_path,
            eeg_data=eeg_arr,
            labels=labels_arr,
            class_names=np.asarray(classes),
            method=np.asarray("game"),
            sampling_rate=np.asarray(int(self._sampling_rate)),
            trial_seconds=np.asarray(float(self._cfg["server"]["trial_seconds"])),
        )
        logger.info(f"Saved game session → {out_path} ({len(labels_arr)} trials)")
        return out_path


    async def _handle_start_calibration(self):
        if self._phase != "collection":
            return
        if len(self._trials) < 6:  
            await self.send({"type": "error", "message": "Not enough trials"})
            return

        self._phase = "calibrating"
        self._emit("phase", "calibrating")
        self._emit("log", f"Starting calibration with {len(self._trials)} trials...")
        logger.info(f"Starting calibration with {len(self._trials)} trials...")
        await self.send({"type": "phase_change", "phase": "calibrating"})

        loop = asyncio.get_event_loop()
        try:
            model_path = await loop.run_in_executor(None, self._run_calibration)
            self._load_classifier(model_path)
            self._phase = "inference"
            await self.send({"type": "calibration_done"})
            await self.send({"type": "phase_change", "phase": "inference"})
            self._emit("phase", "inference")
            self._emit("log", "✓ Calibration done — live inference started")
        except Exception as e:
            logger.exception("Calibration failed")
            self._phase = "collection"
            await self.send({"type": "phase_change", "phase": "collection"})
            await self.send({"type": "error", "message": str(e)})
            self._emit("phase", "collection")
            self._emit("log", f"✗ Calibration failed: {e}")

    async def _handle_start_inference(self):
        if self._clf is None:
            await self.send({"type": "error", "message": "No model loaded"})
            return

        self._phase = "inference"
        await self.send({"type": "phase_change", "phase": "inference"})
        logger.info("Phase → inference")

    def _load_classifier(self, model_path: str):
        sr = self._cfg["preprocessing"]["sampling_rate"]
        self._clf = RealtimeClassifier(
            checkpoint_path = model_path,
            n_channels = self._n_channels,
            n_outputs = 3,
            n_times = self._trial_samples,
            confidence_threshold = self._cfg["live"]["confidence_threshold"],
            preprocessing_config = {
                "sampling_rate": sr,
                "l_freq": 8,
                "h_freq": 30,
                "notch_freq": 50,
            },
        )
        logger.info("✓ Classifier loaded for inference")


    def _run_calibration(self) -> str:
        logger.info("_run_calibration() called")
        left  = sum(1 for t in self._trials if t['label'] == 'Left Hand')
        right = sum(1 for t in self._trials if t['label'] == 'Right Hand')
        rest  = sum(1 for t in self._trials if t['label'] == 'Rest')
        logger.info(f"Trials — Left: {left}, Right: {right}, Rest: {rest}, Total: {left+right+rest}")

          
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

        # EEGNet expects (batch, channels, timepoints) like in RealtimeClassifier.predict()
        X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, C, T)
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
            # Some checkpoints may not fully match the current n_times / architecture.
            model.load_state_dict(model_state, strict=False)
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
