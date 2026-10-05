"""One class order everywhere: 0 Left Hand, 1 Rest, 2 Right Hand."""
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from bcg_core.classifier import RealtimeClassifier
from bcg_core.config_schema import CLASS_INDEX, CLASS_ORDER, LiveConfig, RecordingConfig
from conftest import make_collect_config, server_cfg


def test_order_and_codes():
    assert CLASS_ORDER == ("Left Hand", "Rest", "Right Hand")
    assert CLASS_INDEX == {"Left Hand": 0, "Rest": 1, "Right Hand": 2}


def test_config_defaults_follow_order():
    assert RecordingConfig().labels == list(CLASS_ORDER)
    assert LiveConfig().classes == list(CLASS_ORDER)


def test_graz_labels_cannot_be_reordered():
    with pytest.raises(ValidationError):
        RecordingConfig(labels=["Left Hand", "Right Hand", "Rest"])


def test_classifier_output_index_matches_saved_codes(tmp_path):
    clf = RealtimeClassifier(str(tmp_path / "missing.pth"), n_outputs=3, n_times=512, device="cpu")
    for name, code in CLASS_INDEX.items():
        assert clf.class_names[code] == name


def test_calibration_trains_with_the_same_codes(tmp_path, monkeypatch):
    """The y tensor fed to training must use CLASS_INDEX (before: Right=1, Rest=2)."""
    from bcg_server import bcg_server as mod

    seen = {}
    real_dataset = mod.TensorDataset

    def spy_dataset(X, y):
        seen["y"] = y.tolist()
        return real_dataset(X, y)

    monkeypatch.setattr(mod, "TensorDataset", spy_dataset)
    monkeypatch.setattr(torch, "save", lambda *a, **k: None)

    srv = mod.BCGServer(server_cfg(tmp_path), signals=None)
    labels = ["Left Hand", "Rest", "Right Hand"] * 2
    rng = np.random.default_rng(0)
    srv._trials = [{"label": lb, "eeg": rng.normal(size=(14, 256)).astype(np.float32)} for lb in labels]
    srv._cfg["server"]["trial_seconds"] = 2.0          # 2.0 s * 128 Hz = 256 samples
    srv._cfg["live"]["model_path"] = "models/none.pth"
    srv._run_calibration()
    assert seen["y"] == [CLASS_INDEX[lb] for lb in labels]
