import logging
import time
import numpy as np
from threading import Thread
from PyQt6.QtCore import QObject, pyqtSignal
from bcg_core.cortex_reader import CortexReader

logger = logging.getLogger(__name__)


class EEGWorker(QObject):
    trial_ready  = pyqtSignal(object, str)
    sample_ready = pyqtSignal(object)

    def __init__(self, config):
        super().__init__()
        self._recording = False
        self._buffer    = []
        self._n_channels = config.model.n_channels
        sr = config.preprocessing.sampling_rate

        if hasattr(config, "recording"):
            # DataCollectConfig — data collection mode
            self._simulation    = config.recording.simulation_mode
            self._trial_samples = int(sr * config.recording.trial_seconds)
        else:
            # AppConfig — live inference mode
            self._simulation    = config.live.simulation_mode
            self._trial_samples = int(sr * 4)

        self._reader = CortexReader(config) if not self._simulation else None
        if self._reader is not None:
            # Real headset mode: wire callbacks and start Cortex stream immediately.
            self._reader.on_sample = self.on_eeg_sample
            logger.info("Starting CortexReader (real headset mode)")
            self._reader.start()
        logger.info(f"EEGWorker created — simulation={self._simulation}")

    def can_record_now(self, timeout_sec: float = 0.0) -> bool:
        if self._reader is None:
            return True
        if timeout_sec > 0:
            return self._reader.wait_until_connected(timeout_sec=timeout_sec)
        return self._reader.connected

    def cortex_status(self) -> str:
        if self._reader is None:
            return "Simulation mode enabled"
        return self._reader.get_status_message()

    def start_recording(self, label: str = ""):
        if self._reader is not None and not self._reader.connected:
            logger.error("Cannot start recording: %s", self._reader.get_status_message())
            self._recording = False
            self._buffer = []
            return
        self._buffer = []
        self._recording = True
        self._current_label = label
        logger.info("Recording started")

    def stop_recording(self):
        self._recording = False
        if len(self._buffer) >= self._trial_samples * 0.8:
            buf = self._buffer[:self._trial_samples]
            while len(buf) < self._trial_samples:
                buf.append(buf[-1])
            data = np.array(buf)
            self._buffer = []
            self.trial_ready.emit(data, self._current_label)
            logger.info(f"Trial emitted — shape={data.shape}")
        else:
            logger.warning(f"Buffer too short: {len(self._buffer)} / {self._trial_samples}")
            self._buffer = []
        logger.info("Recording stopped")

    def on_eeg_sample(self, sample: list[float]):
        try:
            self.sample_ready.emit(sample[:self._n_channels])
        except RuntimeError:
            # QObject already deleted (window closed); ignore late samples.
            return
        if not self._recording:
            return
        self._buffer.append(sample[:self._n_channels])

    def start_simulation(self):
        if not self._simulation:
            return
        self._sim_running = True
        self._sim_thread  = Thread(target=self._simulate_loop, daemon=True)
        self._sim_thread.start()
        logger.info("Simulation thread started")

    def stop_simulation(self):
        self._sim_running = False

    def shutdown(self):
        self.stop_simulation()
        if self._reader is not None:
            self._reader.stop()

    def _simulate_loop(self):
        interval = 1.0 / 128
        while self._sim_running:
            try:
                self.on_eeg_sample(self._generate_fake_sample())
            except RuntimeError:
                break
            time.sleep(interval)

    def _generate_fake_sample(self) -> list[float]:
        return [36.0] * self._n_channels


if __name__ == "__main__":
    print("EEGWorker OK")
