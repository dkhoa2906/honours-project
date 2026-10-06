# SPDX-License-Identifier: GPL-3.0-or-later
"""Saved-file schema: keys, dtypes and shapes for Graz and game sessions."""
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from bcg_collect.data_collect_ui import DataCollectionWindow
from bcg_server.bcg_server import BCGServer
from conftest import make_collect_config, server_cfg

GAME_KEYS = {"eeg_data", "labels", "class_names", "method", "sampling_rate", "trial_seconds"}


def fill_server(srv, labels, n_times=512):
    rng = np.random.default_rng(1)
    srv._trials = [{"label": lb, "eeg": rng.normal(size=(14, n_times)).astype(np.float32)} for lb in labels]


def test_game_complete_schema(tmp_path):
    srv = BCGServer(server_cfg(tmp_path, trial_seconds=4.0), signals=None)
    fill_server(srv, ["Left Hand", "Rest", "Right Hand", "Rest"])
    d = np.load(srv._save_game_trials_npz(), allow_pickle=False)
    assert set(d.files) == GAME_KEYS                       # no extra keys for normal sessions
    assert d["eeg_data"].dtype == np.float32 and d["eeg_data"].shape == (4, 512, 14)
    assert d["labels"].dtype == np.int32 and d["labels"].tolist() == [0, 1, 2, 1]
    assert d["class_names"].tolist() == ["Left Hand", "Rest", "Right Hand"]
    assert d["method"].item() == "game" and d["sampling_rate"].item() == 128
    assert d["trial_seconds"].item() == 4.0


def test_game_incomplete_schema_adds_only_two_keys(tmp_path):
    srv = BCGServer(server_cfg(tmp_path, trial_seconds=4.0), signals=None)
    fill_server(srv, ["Left Hand", "Rest"])
    d = np.load(srv._save_game_trials_npz(complete=False), allow_pickle=False)
    assert set(d.files) == GAME_KEYS | {"complete", "n_trials"}
    assert d["complete"].item() is False or d["complete"].item() == False  # noqa: E712
    assert d["n_trials"].item() == 2


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_graz_schema(tmp_path, qapp):
    win = DataCollectionWindow(make_collect_config(str(tmp_path)))
    rng = np.random.default_rng(2)
    for lb in ["Right Hand", "Left Hand", "Rest"]:
        win._on_trial_ready(rng.normal(size=(512, 14)).astype(np.float32), lb)
    win._stop_and_save()
    (path,) = tmp_path.glob("graz_session_*.npz")
    d = np.load(path, allow_pickle=False)
    assert set(d.files) == {"eeg_data", "labels", "class_names"}
    assert d["eeg_data"].dtype == np.float32 and d["eeg_data"].shape == (3, 512, 14)
    assert d["labels"].dtype == np.int32 and d["labels"].tolist() == [2, 0, 1]
    assert d["class_names"].tolist() == ["Left Hand", "Rest", "Right Hand"]
    win.close()
