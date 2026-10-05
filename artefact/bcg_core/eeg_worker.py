import logging
import threading
import time
import numpy as np
from threading import Thread
from PyQt6.QtCore import QObject, pyqtSignal
from bcg_core.cortex_reader import CortexReader

logger = logging.getLogger(__name__)


class EEGWorker(QObject):
    """EEG source shared by all apps: a real headset (CortexReader) or a simulator.

    Signals:
        sample_ready(sample): one sample (first n_channels values) per EEG sample.
        trial_ready(data, label): a finished Graz trial, ``data`` shape (trial_samples, n_channels).
    """

    trial_ready  = pyqtSignal(object, str)
    sample_ready = pyqtSignal(object)

    def __init__(self, config):
        super().__init__()
        self._recording = False
        self._buffer    = []
        # on_eeg_sample runs on the Cortex/simulation thread, start/stop_recording on the Qt thread
        self._buf_lock  = threading.Lock()
        self._sim_running = False
        self._sim_thread  = None
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
        """True if samples are flowing; waits up to ``timeout_sec`` for the headset to connect."""
        if self._reader is None:
            return True
        if timeout_sec > 0:
            return self._reader.wait_until_connected(timeout_sec=timeout_sec)
        return self._reader.connected

    def cortex_status(self) -> str:
        """Human-readable connection status for the UI."""
        if self._reader is None:
            return "Simulation mode enabled"
        return self._reader.get_status_message()

    def ensure_running(self):
        """Restart the sample source if ``shutdown()`` stopped it (a second session in one window)."""
        if self._simulation:
            if not self._sim_running:
                self.start_simulation()
        elif self._reader is not None and self._reader.connection_state == "stopped":
            logger.info("Restarting CortexReader")
            self._reader.start()

    def start_recording(self, label: str = ""):
        """Start collecting samples for one Graz trial with the given class label."""
        if self._reader is not None and not self._reader.connected:
            logger.error("Cannot start recording: %s", self._reader.get_status_message())
            with self._buf_lock:
                self._recording = False
                self._buffer = []
            return
        with self._buf_lock:
            self._buffer = []
            self._recording = True
        self._current_label = label
        logger.info("Recording started")

    def stop_recording(self):
        """Finish the Graz trial and emit ``trial_ready``.

        Keeps the trial if it has at least 80 % of ``trial_samples`` samples, otherwise
        drops it. A short trial is padded by repeating its last sample; a long one is
        trimmed to its FIRST ``trial_samples`` samples. (The game server trims to the
        LAST samples instead; the two are deliberately not aligned.)
        """
        with self._buf_lock:
            self._recording = False
            buffer, self._buffer = self._buffer, []
        if len(buffer) >= self._trial_samples * 0.8:
            buf = buffer[:self._trial_samples]
            while len(buf) < self._trial_samples:
                buf.append(buf[-1])
            data = np.array(buf)
            self.trial_ready.emit(data, self._current_label)
            logger.info(f"Trial emitted — shape={data.shape}")
        else:
            logger.warning(f"Buffer too short: {len(buffer)} / {self._trial_samples}")
        logger.info("Recording stopped")

    def on_eeg_sample(self, sample: list[float]):
        """Receive one EEG sample from the reader or the simulator."""
        try:
            self.sample_ready.emit(sample[:self._n_channels])
        except RuntimeError:
            # QObject already deleted (window closed); ignore late samples.
            return
        with self._buf_lock:
            if self._recording:
                self._buffer.append(sample[:self._n_channels])

    def start_simulation(self):
        """Start emitting fixed-value samples at 128 Hz (simulation mode only)."""
        if not self._simulation:
            return
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._sim_running = True      # a stop was requested but the thread has not exited yet
            return
        self._sim_running = True
        self._sim_thread  = Thread(target=self._simulate_loop, daemon=True)
        self._sim_thread.start()
        logger.info("Simulation thread started")

    def stop_simulation(self):
        """Stop the simulator and wait briefly for its thread to exit."""
        self._sim_running = False
        t = self._sim_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)

    def shutdown(self):
        """Stop the sample source (simulator or Cortex connection)."""
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
