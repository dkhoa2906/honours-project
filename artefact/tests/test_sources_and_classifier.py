"""Sample source restart (Graz second session), classifier failure handling, config paths."""
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from bcg_collect.data_collect_ui import DataCollectionWindow
from bcg_core.classifier import RealtimeClassifier
from bcg_core.eeg_worker import EEGWorker
from bcg_core.paths import ROOT, resolve_path
from conftest import make_collect_config


def count_samples(worker, seconds):
    """Samples produced by the simulator in ``seconds`` (counted at the source: Qt signal
    slots are queued to the main thread and need an event loop, which tests do not run)."""
    n = []
    make = worker._generate_fake_sample
    worker._generate_fake_sample = lambda: (n.append(1), make())[1]
    time.sleep(seconds)
    worker._generate_fake_sample = make
    return len(n)


def test_simulator_restarts_after_shutdown_without_doubling_rate(tmp_path):
    worker = EEGWorker(make_collect_config(str(tmp_path)))
    worker.start_simulation()
    first = count_samples(worker, 0.5)
    worker.shutdown()
    worker.ensure_running()                       # what _start_session does for a second session
    second = count_samples(worker, 0.5)
    worker.shutdown()
    assert 30 < first < 90 and 30 < second < 90   # about 64 per 0.5 s at 128 Hz, not 128


def test_ensure_running_is_a_noop_while_running(tmp_path):
    worker = EEGWorker(make_collect_config(str(tmp_path)))
    worker.start_simulation()
    thread = worker._sim_thread
    worker.ensure_running()
    assert worker._sim_thread is thread
    worker.shutdown()


def test_graz_window_can_start_a_second_session(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = DataCollectionWindow(make_collect_config(str(tmp_path)))
    rng = np.random.default_rng(0)
    win._on_trial_ready(rng.normal(size=(512, 14)).astype(np.float32), "Rest")
    win._stop_and_save()
    assert not win._eeg_worker._sim_running
    win._start_session()
    assert win._eeg_worker._sim_running
    win.timer.stop()
    win._eeg_worker.shutdown()
    win.close()


@pytest.fixture(scope="module")
def clf(tmp_path_factory):
    return RealtimeClassifier(str(tmp_path_factory.mktemp("w") / "missing.pth"),
                              n_outputs=3, n_times=512, device="cpu")


def test_classifier_failure_is_not_reported_as_rest(clf, monkeypatch):
    def broken(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(clf, "model", broken)
    label, conf = clf.predict(np.zeros((14, 512), dtype=np.float32))
    assert label is None and conf == 0.0


def test_classifier_normal_prediction(clf):
    rng = np.random.default_rng(3)
    label, conf = clf.predict(rng.normal(size=(14, 512)).astype(np.float32))
    assert label in clf.class_names and 0.0 <= conf <= 100.0


def test_relative_paths_resolve_against_repository_root():
    assert resolve_path("recordings") == ROOT / "recordings"
    assert resolve_path(ROOT / "x") == ROOT / "x"
