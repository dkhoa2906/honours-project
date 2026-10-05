"""80 % rule, pad (repeat last sample) and trim, for BOTH modules.

Graz   (EEGWorker.stop_recording):  keeps the FIRST 512 samples, accepts >= 410 (80 % of 512 = 409.6).
Server (BCGServer._handle_trial_end): keeps the LAST 512 samples, accepts >= 409 (int(409.6)).
Sample k carries the value k on every channel, so the kept window can be read off the data.
"""
import numpy as np
import pytest

from bcg_core.eeg_worker import EEGWorker
from bcg_server.bcg_server import BCGServer
from conftest import make_collect_config, server_cfg

TARGET = 512


# ---------------------------------------------------------------- Graz
def graz_trial(tmp_path, n_samples):
    """Record n_samples through the Graz EEGWorker; return the emitted (512, 14) array or None."""
    worker = EEGWorker(make_collect_config(str(tmp_path)))
    emitted = []
    worker.trial_ready.connect(lambda data, label: emitted.append((data, label)))
    worker.start_recording("Left Hand")
    for k in range(n_samples):
        worker.on_eeg_sample([float(k)] * 14)
    worker.stop_recording()
    return emitted[0][0] if emitted else None


def test_graz_exact_length(tmp_path):
    data = graz_trial(tmp_path, 512)
    assert data.shape == (TARGET, 14) and data[0, 0] == 0 and data[-1, 0] == 511


def test_graz_long_window_keeps_first_512(tmp_path):
    data = graz_trial(tmp_path, 600)
    assert data.shape == (TARGET, 14)
    assert data[0, 0] == 0 and data[-1, 0] == 511          # FIRST 512


def test_graz_short_window_is_padded_with_last_sample(tmp_path):
    data = graz_trial(tmp_path, 450)
    assert data.shape == (TARGET, 14)
    assert data[449, 0] == 449 and (data[450:, 0] == 449).all()   # repeat, not zeros


@pytest.mark.parametrize("n, kept", [(410, True), (409, False), (300, False), (0, False)])
def test_graz_threshold(tmp_path, n, kept):
    assert (graz_trial(tmp_path, n) is not None) is kept


# ---------------------------------------------------------------- Game server
@pytest.fixture
def server(tmp_path):
    return BCGServer(server_cfg(tmp_path, trial_seconds=4.0), signals=None)


def server_trial(srv, lt, n_samples, label="Right Hand", preroll=10):
    """Feed samples around a trial_start/trial_end pair; return the stored (14, 512) array or None."""
    k = srv._sample_counter
    for _ in range(preroll):
        srv._on_sample(np.full(14, k, dtype=np.float32)); k += 1
    lt.run(srv._handle_trial_start({"label": label}))
    first = k
    for _ in range(n_samples):
        srv._on_sample(np.full(14, k, dtype=np.float32)); k += 1
    lt.run(srv._handle_trial_end({"label": label}))
    return (srv._trials[-1]["eeg"] if srv._trials else None), first


def test_server_exact_length(server, loop_thread):
    eeg, first = server_trial(server, loop_thread, 512)
    assert eeg.shape == (14, TARGET) and eeg[0, 0] == first and eeg[0, -1] == first + 511


def test_server_long_window_keeps_last_512(server, loop_thread):
    eeg, first = server_trial(server, loop_thread, 600)
    assert eeg.shape == (14, TARGET)
    assert eeg[0, 0] == first + 88 and eeg[0, -1] == first + 599      # LAST 512


def test_server_short_window_is_padded_with_last_sample(server, loop_thread):
    eeg, first = server_trial(server, loop_thread, 450)
    assert eeg.shape == (14, TARGET)
    assert eeg[0, 449] == first + 449 and (eeg[0, 450:] == first + 449).all()


@pytest.mark.parametrize("n, kept", [(409, True), (408, False), (300, False), (0, False)])
def test_server_threshold(server, loop_thread, n, kept):
    eeg, _ = server_trial(server, loop_thread, n)
    assert (eeg is not None) is kept
    if not kept:
        assert server._rejected["too_short"] == 1 and server._current_trial is None


def test_server_rejects_wrong_channel_count(tmp_path, loop_thread):
    srv = BCGServer(server_cfg(tmp_path, trial_seconds=4.0), signals=None)
    k = 0
    loop_thread.run(srv._handle_trial_start({"label": "Rest"}))
    for _ in range(512):
        srv._on_sample(np.zeros(10, dtype=np.float32))          # 10 channels instead of 14
    loop_thread.run(srv._handle_trial_end({"label": "Rest"}))
    assert srv._trials == [] and srv._rejected["wrong_shape"] == 1


def test_trim_sides_differ_between_modules(tmp_path, server, loop_thread):
    graz = graz_trial(tmp_path, 600)
    eeg, first = server_trial(server, loop_thread, 600)
    assert graz[0, 0] == 0                       # Graz: window starts at the first sample
    assert eeg[0, 0] == first + 88               # server: window starts 88 samples later
