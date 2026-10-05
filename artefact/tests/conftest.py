"""Shared test setup: headless Qt, repo root on sys.path, helpers for the server tests."""
import asyncio
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from bcg_core.config_schema import AppConfig, DataCollectConfig


def make_app_config(**live) -> AppConfig:
    return AppConfig.model_validate({
        "cortex_api": {"client_id": "test", "client_secret": "test"},
        "live": {"simulation_mode": True, **live},
    })


def make_collect_config(save_path: str, **recording) -> DataCollectConfig:
    return DataCollectConfig.model_validate({
        "cortex_api": {"client_id": "test", "client_secret": "test"},
        "recording": {"simulation_mode": True, "save_path": save_path, **recording},
    })


def server_cfg(save_dir, trial_seconds: float = 0.5, port: int = 0) -> dict:
    """Raw dict config for BCGServer (same layout as config/server_conf.json)."""
    return {
        "preprocessing": {"sampling_rate": 128},
        "model": {"n_channels": 14},
        "live": {"step_samples": 64, "simulation_mode": True,
                 "classes": ["Left Hand", "Rest", "Right Hand"],
                 "confidence_threshold": 0.5, "model_path": "models/none.pth"},
        "server": {"host": "localhost", "port": port, "trial_seconds": trial_seconds,
                   "calibration_epochs": 1, "save_dir": str(save_dir)},
    }


class LoopThread:
    """An asyncio loop in a background thread, so sync tests can drive BCGServer coroutines."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)


@pytest.fixture
def loop_thread():
    lt = LoopThread()
    yield lt
    lt.close()
