import asyncio
import json
import logging
import threading
from pathlib import Path


from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFrame
)
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QFont

from bcg_core.config_schema import AppConfig
from bcg_core.eeg_worker import EEGWorker
from bcg_server import BCGServer

logger = logging.getLogger(__name__)

COLORS = {
    "bg":       "#1a1a1a",
    "surface":  "#2d2d2d",
    "surface2": "#383838",
    "text":     "#e0e0e0",
    "muted":    "#888888",
    "green":    "#2E8B57",
    "yellow":   "#FFD700",
    "red":      "#DC143C",
}


class ServerSignals(QObject):
    log = pyqtSignal(str)
    phase = pyqtSignal(str)
    headset = pyqtSignal(bool)
    trial_count = pyqtSignal(int)
    clients = pyqtSignal(int)
    eeg = pyqtSignal(str)



class BCGServerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCG Game Server")
        self.setMinimumSize(700, 540)

        self._signals = ServerSignals()
        self._signals.log.connect(self._log_line)
        self._signals.phase.connect(self._update_phase)
        self._signals.headset.connect(self._update_headset)
        self._signals.trial_count.connect(self._update_trials)
        self._signals.clients.connect(self._update_clients)
        self._signals.eeg.connect(self._update_eeg)

        self._server = None
        self._eeg_worker = None
        
        self._server_thread = None
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("BCG Game Server")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self._lb_phase = QLabel("● Idle")
        self._lb_phase.setFont(QFont("Segoe UI", 11))
        self._lb_phase.setStyleSheet(f"color: {COLORS['muted']};")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self._lb_phase)
        root.addLayout(hdr)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #383838;")
        root.addWidget(line)

        # Status cards
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self._card_headset = self._make_card("Headset", "○ Disconnected", COLORS["muted"])
        self._card_clients = self._make_card("Browser", "○ No client",    COLORS["muted"])
        self._card_trials  = self._make_card("Trials",  "0",               COLORS["text"])
        self._card_phase   = self._make_card("Phase",   "—",               COLORS["muted"])
        for card in [self._card_headset, self._card_clients, self._card_trials, self._card_phase]:
            cards.addWidget(card["widget"])
        root.addLayout(cards)

        # WS address
        addr_row = QHBoxLayout()
        lb_addr = QLabel("ws://")
        lb_addr.setFont(QFont("Segoe UI", 11))
        lb_addr.setStyleSheet(f"color: {COLORS['muted']};")
        self._input_addr = QLineEdit("localhost:8765")
        self._input_addr.setFixedHeight(36)
        self._input_addr.setFont(QFont("Segoe UI", 11))
        addr_row.addWidget(lb_addr)
        addr_row.addWidget(self._input_addr)
        root.addLayout(addr_row)

        # Log
        lb_log = QLabel("Server Log")
        lb_log.setFont(QFont("Segoe UI", 9))
        lb_log.setStyleSheet(f"color: {COLORS['muted']};")
        root.addWidget(lb_log)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 9))
        self._log.setStyleSheet(
            "QTextEdit { background:#222; color:#aaa; border:none; border-radius:6px; padding:6px; }"
        )
        root.addWidget(self._log, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self._btn_start = QPushButton("▶ Start Server")
        self._btn_start.setFixedHeight(42)
        self._btn_start.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._btn_start.clicked.connect(self._start_server)
        self._btn_stop = QPushButton("■ Stop")
        self._btn_stop.setFixedHeight(42)
        self._btn_stop.setFont(QFont("Segoe UI", 11))
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_server)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_stop)
        root.addLayout(btn_row)

        # EEG debug line
        self._lb_eeg = QLabel("EEG: —")
        self._lb_eeg.setFont(QFont("Segoe UI", 9))
        self._lb_eeg.setStyleSheet(f"color: {COLORS['muted']};")
        root.addWidget(self._lb_eeg)

    def _make_card(self, title: str, value: str, color: str) -> dict:
        card = QWidget()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(4)
        lb_title = QLabel(title)
        lb_title.setFont(QFont("Segoe UI", 9))
        lb_title.setStyleSheet(f"color: {COLORS['muted']};")
        lb_val = QLabel(value)
        lb_val.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        lb_val.setStyleSheet(f"color: {color};")
        cl.addWidget(lb_title)
        cl.addWidget(lb_val)
        return {"widget": card, "label": lb_val}

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color:{COLORS['bg']}; color:{COLORS['text']}; }}
            #card {{ background-color:{COLORS['surface']}; border-radius:12px; }}
            #card QLabel {{ background:transparent; }}
            QLineEdit {{
                background:{COLORS['surface2']}; color:{COLORS['text']};
                border:1px solid #555; border-radius:8px; padding:0 10px;
            }}
            QPushButton {{
                background-color:{COLORS['surface2']}; color:{COLORS['text']};
                border:1px solid #555; border-radius:8px; padding:0 20px;
            }}
            QPushButton:hover    {{ background-color:#484848; }}
            QPushButton:disabled {{ color:#555; border-color:#444; }}
        """)

    # ---> Signals handler
    def _log_line(self, msg: str):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _update_phase(self, phase: str):
        labels = {
            "collection":  ("● Collecting",  COLORS["yellow"]),
            "calibrating": ("⟳ Calibrating", COLORS["yellow"]),
            "inference":   ("⚡ Inference",   COLORS["green"]),
        }
        text, color = labels.get(phase, ("● Idle", COLORS["muted"]))
        self._lb_phase.setText(text)
        self._lb_phase.setStyleSheet(f"color: {color};")
        self._card_phase["label"].setText(phase.capitalize())
        self._card_phase["label"].setStyleSheet(f"color: {color};")

    def _update_headset(self, connected: bool):
        if connected:
            self._card_headset["label"].setText("● Connected")
            self._card_headset["label"].setStyleSheet(f"color: {COLORS['green']};")
        else:
            self._card_headset["label"].setText("○ Disconnected")
            self._card_headset["label"].setStyleSheet(f"color: {COLORS['red']};")

    def _update_trials(self, count: int):
        self._card_trials["label"].setText(str(count))

    def _update_clients(self, count: int):
        if count > 0:
            self._card_clients["label"].setText(f"● {count} connected")
            self._card_clients["label"].setStyleSheet(f"color: {COLORS['green']};")
        else:
            self._card_clients["label"].setText("○ No client")
            self._card_clients["label"].setStyleSheet(f"color: {COLORS['muted']};")
    
    def _update_eeg(self, text: str):
        self._lb_eeg.setText(f"EEG: {text}")
    # <---


    # ---> Server control
    def _start_server(self):
        addr = self._input_addr.text().strip()
        host, port = addr.split(":") if ":" in addr else ("localhost", "8765")

        cfg_path = Path(__file__).parent.parent / "config" / "server_conf.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        cfg["server"]["host"] = host
        cfg["server"]["port"] = int(port)

        # Build typed config and server
        app_cfg = AppConfig.model_validate(cfg)
        self._server = BCGServer(cfg, self._signals)

        # EEGWorker as EEG source (simulation or real)
        self._eeg_worker = EEGWorker(app_cfg)
        self._eeg_worker.sample_ready.connect(self._server._on_sample)

        if app_cfg.live.simulation_mode:
            self._eeg_worker.start_simulation()

        # Start asyncio server on background thread
        def run_loop():
            asyncio.run(self._server.run())

        self._server_thread = threading.Thread(target=run_loop, daemon=True)
        self._server_thread.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._input_addr.setEnabled(False)


    def _stop_server(self):
        self._log_line("Server stopped (restart app to reuse port).")
        self._btn_stop.setEnabled(False)
        self._update_phase("idle")
    # <---