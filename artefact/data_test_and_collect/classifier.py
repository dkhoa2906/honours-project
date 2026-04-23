import logging
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from scipy.signal import butter, filtfilt, iirnotch

try:
    from braindecode.models import EEGNetv4 as EEGNet
except ImportError:
    from braindecode.models import EEGNet

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    def __init__(self, sampling_rate=128, l_freq=8, h_freq=30, notch_freq=50):
        nyq = sampling_rate / 2
        self.b_bp, self.a_bp = butter(5, [l_freq/nyq, h_freq/nyq], btype="band")
        self.b_n,  self.a_n  = iirnotch(notch_freq, Q=30, fs=sampling_rate)

    def process(self, eeg: np.ndarray) -> np.ndarray:
        x = eeg.copy().astype(np.float32)
        for ch in range(x.shape[0]):
            x[ch] = filtfilt(self.b_bp, self.a_bp, x[ch])
            x[ch] = filtfilt(self.b_n,  self.a_n,  x[ch])
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1,  keepdims=True) + 1e-8
        return ((x - mu) / std).astype(np.float32)


class RealtimeClassifier:
    def __init__(
        self,
        checkpoint_path: str,
        n_channels: int = 14,
        n_outputs: int = 3,
        n_times: int = 512,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        debug: bool = False,
        preprocessing_config: dict = None,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        cfg = preprocessing_config or {"sampling_rate": 128, "l_freq": 8, "h_freq": 30, "notch_freq": 50}
        self.preprocessor = EEGPreprocessor(**cfg)
        self.model = EEGNet(
            n_chans=n_channels, n_outputs=n_outputs,
            n_times=n_times, drop_prob=0.5
        ).to(self.device)
        self.model.eval()

        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            raw   = torch.load(ckpt, map_location=self.device)
            state = raw.get("model_state_dict") or raw.get("state_dict") or raw
            fixed = {k.replace(".parametrizations.weight.original", ".weight"): v
                     for k, v in state.items()}
            self.model.load_state_dict(fixed, strict=False)
            logger.info(f"Loaded checkpoint: {ckpt}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt} — using random weights")

        self.debug = debug
        self.class_names = (
            ["Left Hand", "Right Hand"]         if n_outputs == 2 else
            ["Left Hand", "Rest", "Right Hand"] if n_outputs == 3 else
            [f"Class {i}" for i in range(n_outputs)]
        )

    def predict(self, eeg_window: np.ndarray) -> Tuple[str, float]:
        try:
            x = self.preprocessor.process(eeg_window)
            x = torch.tensor(x[np.newaxis], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
            idx    = int(np.argmax(probs))
            s      = np.sort(probs)[::-1]
            margin = float(s[0] - s[1])
            if self.debug:
                logger.info(" | ".join(f"{self.class_names[i]}={probs[i]:.3f}"
                                       for i in range(len(self.class_names)))
                            + f" | margin={margin:.3f}")
            if margin < 0.2:
                return "Rest", margin * 100.0
            return self.class_names[idx], float(probs[idx]) * 100.0
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Rest", 0.0
