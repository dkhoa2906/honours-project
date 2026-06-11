import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import deque

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from bcg_core.config_schema import AppConfig
from bcg_core.eeg_worker import EEGWorker
from bcg_core.classifier import RealtimeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLORS = {
    "bg":      "#1a1a1a",
    "surface": "#2d2d2d",
    "surface2":"#383838",
    "text":    "#e0e0e0",
    "muted":   "#888888",
    "green":   "#2E8B57",
    "yellow":  "#FFD700",
}


class MILiveWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self._config       = config
        self._live         = config.live
        self._classes      = self._live.classes
        self._class_colors = self._live.colors

        self._buffer     = deque(maxlen=int(config.preprocessing.sampling_rate * 4))
        self._step_count = 0
        self._running    = False
        self._last_pred  = None

        self._classifier = RealtimeClassifier(
            checkpoint_path      = self._live.model_path,
            n_channels           = config.model.n_channels,
            n_outputs            = self._live.n_outputs,
            n_times              = int(config.preprocessing.sampling_rate * 4),
            confidence_threshold = self._live.confidence_threshold,
            preprocessing_config = {
                "sampling_rate": config.preprocessing.sampling_rate,
                "l_freq": 8, "h_freq": 30, "notch_freq": 50
            }
        )

        self.setWindowTitle("MI Live Monitor")
        self.setMinimumSize(860, 620)
        self._build_ui()
        self._apply_theme()

        self._eeg_worker = EEGWorker(config)
        self._eeg_worker.sample_ready.connect(self._on_sample)
        if self._live.simulation_mode:
            self._eeg_worker.start_simulation()

        self._timer = QTimer()
        self._timer.timeout.connect(self._maybe_predict)



    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel("Motor Imagery · Live Monitor")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self._lb_status = QLabel("● Idle")
        self._lb_status.setFont(QFont("Segoe UI", 11))
        self._lb_status.setStyleSheet("color: #888;")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self._lb_status)
        root.addLayout(hdr)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #383838;")
        root.addWidget(line)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self._conf_bars    = {}
        self._conf_labels  = {}
        self._card_widgets = {}

        for cls in self._classes:
            card = QWidget()
            card.setObjectName("card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 16, 16, 16)
            cl.setSpacing(8)

            lb_name = QLabel(cls)
            lb_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb_name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            lb_name.setStyleSheet(f"color: {self._class_colors.get(cls, '#e0e0e0')};")

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            bar.setStyleSheet(f"""
                QProgressBar {{ background:#383838; border-radius:7px; border:none; }}
                QProgressBar::chunk {{ background:{self._class_colors.get(cls,'#888')}; border-radius:7px; }}
            """)

            lb_conf = QLabel("0%")
            lb_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb_conf.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
            lb_conf.setStyleSheet("color: #e0e0e0;")

            cl.addWidget(lb_name)
            cl.addWidget(bar)
            cl.addWidget(lb_conf)

            self._conf_bars[cls]    = bar
            self._conf_labels[cls]  = lb_conf
            self._card_widgets[cls] = card
            cards_row.addWidget(card)

        root.addLayout(cards_row)

        self._lb_cue = QLabel("—")
        self._lb_cue.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lb_cue.setFont(QFont("Segoe UI", 38, QFont.Weight.Bold))
        self._lb_cue.setStyleSheet("color: #e0e0e0;")
        self._lb_cue.setFixedHeight(70)
        root.addWidget(self._lb_cue)

        buf_row = QHBoxLayout()
        lb_buf = QLabel("Buffer")
        lb_buf.setFont(QFont("Segoe UI", 9))
        lb_buf.setStyleSheet("color: #888;")
        self._buf_bar = QProgressBar()
        self._buf_bar.setRange(0, int(self._config.preprocessing.sampling_rate * 4))
        self._buf_bar.setValue(0)
        self._buf_bar.setFixedHeight(6)
        self._buf_bar.setTextVisible(False)
        self._buf_bar.setStyleSheet("""
            QProgressBar { background:#383838; border-radius:3px; border:none; }
            QProgressBar::chunk { background:#FFD700; border-radius:3px; }
        """)
        buf_row.addWidget(lb_buf)
        buf_row.addWidget(self._buf_bar)
        root.addLayout(buf_row)

        lb_log = QLabel("Prediction Log")
        lb_log.setFont(QFont("Segoe UI", 9))
        lb_log.setStyleSheet("color: #888;")
        root.addWidget(lb_log)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setFixedHeight(140)
        self._log.setStyleSheet("""
            QTextEdit { background:#222; color:#aaa; border:none; border-radius:6px; padding:6px; }
        """)
        root.addWidget(self._log)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setFixedHeight(42)
        self._btn_start.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._btn_start.clicked.connect(self._start)
        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setFixedHeight(42)
        self._btn_stop.setFont(QFont("Segoe UI", 11))
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color:{COLORS['bg']}; color:{COLORS['text']}; }}
            #card {{ background-color:{COLORS['surface']}; border-radius:12px; }}
            #card QLabel {{ background:transparent; }}
            QPushButton {{ background-color:#383838; color:{COLORS['text']}; border:1px solid #555; border-radius:8px; padding:0 20px; }}
            QPushButton:hover {{ background-color:#484848; }}
            QPushButton:disabled {{ color:#555; border-color:#444; }}
        """)



    def _start(self):
        self._running = True
        self._step_count = 0
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lb_status.setText("● Streaming")
        self._lb_status.setStyleSheet(f"color:{COLORS['green']};")
        self._timer.start(50)
        self._log_line("Session started")

    def _stop(self):
        self._running = False
        self._timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lb_status.setText("● Stopped")
        self._lb_status.setStyleSheet(f"color:{COLORS['muted']};")
        self._reset_cards()
        self._lb_cue.setText("—")
        self._log_line("Session stopped")

    def closeEvent(self, event):
        self._eeg_worker.shutdown()
        super().closeEvent(event)



    def _on_sample(self, sample):
        self._buffer.append(sample)
        self._step_count += 1
        self._buf_bar.setValue(len(self._buffer))

    def _maybe_predict(self):
        if not self._running:
            return
        win_size = int(self._config.preprocessing.sampling_rate * 4)
        if len(self._buffer) < win_size:
            return
        if self._step_count < self._live.step_samples:
            return
        self._step_count = 0
        window = np.array(self._buffer, dtype=np.float32).T
        try:
            pred, conf = self._classifier.predict(window)
            self._update_ui(pred, conf)
        except Exception as e:
            logger.error(f"Predict error: {e}")



    def _update_ui(self, pred: str, conf: float):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")

        for cls in self._classes:
            self._conf_bars[cls].setValue(0)
            self._conf_labels[cls].setText("0%")
            self._card_widgets[cls].setStyleSheet(
                "#card { background:#2d2d2d; border-radius:12px; } #card QLabel { background:transparent; }"
            )

        if pred in self._conf_bars:
            self._conf_bars[pred].setValue(int(conf))
            self._conf_labels[pred].setText(f"{conf:.0f}%")
            color = self._class_colors.get(pred, "#888")
            self._card_widgets[pred].setStyleSheet(
                f"#card {{ background:#2d2d2d; border-radius:12px; border:2px solid {color}; }} #card QLabel {{ background:transparent; }}"
            )

        if pred == "Rest":
            self._lb_cue.setText("●  Rest")
            self._lb_cue.setStyleSheet(f"color:{self._class_colors.get('Rest','#808080')};")
        else:
            arrow = "←" if pred == "Left Hand" else "→"
            self._lb_cue.setText(f"{arrow}  {pred}")
            self._lb_cue.setStyleSheet(f"color:{self._class_colors.get(pred,'#e0e0e0')};")

        self._log_line(f"[{ts}]  {pred:<12}  {conf:.1f}%")
        self._last_pred = pred

    def _reset_cards(self):
        for cls in self._classes:
            self._conf_bars[cls].setValue(0)
            self._conf_labels[cls].setText("0%")
            self._card_widgets[cls].setStyleSheet(
                "#card { background:#2d2d2d; border-radius:12px; } #card QLabel { background:transparent; }"
            )

    def _log_line(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())


if __name__ == "__main__":
    import json
    with open("config/milive_conf.json") as f:
        config = AppConfig.model_validate(json.load(f))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MILiveWindow(config)
    win.show()
    sys.exit(app.exec())
