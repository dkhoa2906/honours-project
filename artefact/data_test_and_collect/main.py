import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QLabel, QPushButton, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from config_schema import AppConfig, DataCollectConfig


STYLE = """
QWidget {
    background-color: #1a1a1a;
    color: #e0e0e0;
}
QPushButton {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #555;
    border-radius: 10px;
    padding: 16px 32px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover { background-color: #3a3a3a; border-color: #888; }
QPushButton:pressed { background-color: #444; }
"""


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MI-BCI Launcher")
        self.setFixedSize(420, 320)
        self.setStyleSheet(STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        title = QLabel("MI-BCI")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Motor Imagery Brain-Computer Interface")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888;")

        btn_collect = QPushButton("📊  Data Collection")
        btn_collect.setFixedHeight(60)
        btn_collect.clicked.connect(self._launch_collect)

        btn_live = QPushButton("🧠  Live Inference")
        btn_live.setFixedHeight(60)
        btn_live.clicked.connect(self._launch_live)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(btn_collect)
        layout.addWidget(btn_live)
        layout.addStretch()

    def _launch_collect(self):
        from data_collect_ui import DataCollectionWindow
        cfg_path = "config/datacollect_conf.json"
        with open(cfg_path) as f:
            config = DataCollectConfig.model_validate(json.load(f))
        self._child = DataCollectionWindow(config)
        self._child.show()
        self.hide()

    def _launch_live(self):
        from mi_live_ui import MILiveWindow
        cfg_path = "config/milive_conf.json"
        with open(cfg_path) as f:
            config = AppConfig.model_validate(json.load(f))
        self._child = MILiveWindow(config)
        self._child.show()
        self.hide()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec())