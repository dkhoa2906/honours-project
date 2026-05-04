
import logging
import sys
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QFont
from bcg_core.config_schema import DataCollectConfig
from bcg_core.eeg_worker import EEGWorker

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DataCollectionWindow(QMainWindow):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._load_config()

        # Session state
        self.current_trial = 0
        self.phase = "idle"
        self.phase_start = 0.0
        self.phase_duration = 0.0
        self.trial_order = []
        self.collected_eeg = []
        self.collected_labels = []
        self._in_break = False

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

        self.setWindowTitle("BCG EEG Recorder for Motor Imagery")
        self.setMinimumSize(900, 650)
        self._build_ui()
        self._apply_theme()

        self._eeg_worker = EEGWorker(self._config)
        self._eeg_worker.trial_ready.connect(self._on_trial_ready)
        if self._config.recording.simulation_mode:
            self._eeg_worker.start_simulation()

        # Connect buttons
        self.btn_start.clicked.connect(self._start_session)
        self.btn_save.clicked.connect(self._stop_and_save)
        self.btn_skip_break.clicked.connect(self._skip_break)
        self.btn_folder.clicked.connect(self._open_folder)

    # FUNCTION GROUP 1: SETUP FUNCTIONS
    def _load_config(self):
        rec = self._config.recording
        pre = self._config.preprocessing

        self.SAMPLING_RATE    = pre.sampling_rate
        self.TRIAL_SECONDS    = rec.trial_seconds
        self.PREPARE_SECONDS  = rec.prepare_seconds
        self.REST_SECONDS     = rec.rest_seconds
        self.BREAK_SECONDS    = rec.break_seconds
        self.N_BLOCKS         = rec.n_blocks
        self.TRIALS_PER_BLOCK = rec.trials_per_block
        self.TRIAL_SAMPLES    = int(self.SAMPLING_RATE * self.TRIAL_SECONDS)

        self.CLASSES          = rec.labels
        self.N_CLASSES        = len(self.CLASSES)
        self.TRIALS_PER_CLASS = self.TRIALS_PER_BLOCK * self.N_BLOCKS

        self.COLORS = {
                **rec.colors,
                "bg":      "#1a1a1a",
                "surface": "#2d2d2d",
                "text":    "#e0e0e0",
            }

        self.SAVE_PATH = rec.save_path


    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        self.lay = QVBoxLayout(central)
        self.lay.setContentsMargins(40, 24, 40, 24)
        self.lay.setSpacing(12)

        # 1st layer: row1 ---->
        self.lb_block = QLabel("Block — / —")
        self.lb_trial = QLabel("Trial — / —")
        self.lb_trial.setFont(QFont("Segoe UI", 10))
        self.lb_block.setFont(QFont("Segoe UI", 10))
        row1 = QHBoxLayout()
        row1.addWidget(self.lb_block)
        row1.addStretch()
        row1.addWidget(self.lb_trial)

        # 2nd layer: pgbar_total ---->
        self.pgbar_total = QProgressBar()
        self.pgbar_total.setRange(0, self.TRIALS_PER_CLASS * self.N_CLASSES)
        self.pgbar_total.setValue(0)
        self.pgbar_total.setFixedHeight(8)
        self.pgbar_total.setTextVisible(False)

        # 3rd layer: instr_box ---->
        instr_box = QWidget()
        instr_box.setObjectName("instrBox")
        
        instr_lay = QVBoxLayout(instr_box)
        instr_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instr_lay.setContentsMargins(24, 20, 24, 20)
        instr_lay.setSpacing(6)

        self.lb_cue = QLabel("Press START to begin")
        self.lb_cue.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lb_cue.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))

        self.lb_arrow = QLabel("")
        self.lb_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lb_arrow.setFont(QFont("Segoe UI", 72))

        self.lb_phase = QLabel("")
        self.lb_phase.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lb_phase.setFont(QFont("Segoe UI", 13))

        instr_lay.addWidget(self.lb_cue)
        instr_lay.addWidget(self.lb_arrow)
        instr_lay.addWidget(self.lb_phase)

        # 4th layer: pgbar_trial ---->
        self.pgbar_trial = QProgressBar()
        self.pgbar_trial.setRange(0, 100)
        self.pgbar_trial.setValue(0)
        self.pgbar_trial.setFixedHeight(12)
        self.pgbar_trial.setTextVisible(False)

        # 5th layer: row5 ---->
        """
        row5 (QHBoxLayout)
        └─ box (QWidget)
            └─ box_lay (QVBoxLayout)
                ├─ lb_name (QLabel)
                └─ lb_count (QLabel)
        """
        self.count_labels = {}
        self.class_boxes = {}

        row5 = QHBoxLayout()
        row5.setSpacing(12)
        
        for cls in self.CLASSES:
            lb_name = QLabel(cls)
            lb_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb_name.setFont(QFont("Segoe UI", 11))

            lb_count = QLabel(f"0 / {self.TRIALS_PER_CLASS}")
            lb_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb_count.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
            self.count_labels[cls] = lb_count
            
            box = QWidget()
            box.setObjectName("classBox")
            box.setMinimumHeight(90)

            box_lay = QVBoxLayout(box)
            box_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box_lay.setSpacing(4)
            box_lay.addWidget(lb_name)
            box_lay.addWidget(lb_count)

            self.class_boxes[cls] = box
            row5.addWidget(box)

        # 6th layer: buttons ---->
        row6 = QHBoxLayout()
        row6.setSpacing(10)

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setFixedHeight(46)
        self.btn_start.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))

        self.btn_skip_break = QPushButton("⏭  Skip Break")
        self.btn_skip_break.setFixedHeight(46)
        self.btn_skip_break.setFont(QFont("Segoe UI", 12))
        self.btn_skip_break.setDisabled(False)

        self.btn_save = QPushButton("■  STOP & SAVE")
        self.btn_save.setFixedHeight(46)
        self.btn_save.setFont(QFont("Segoe UI", 12))
        self.btn_save.setEnabled(False)

        self.btn_folder = QPushButton("📁  Recordings")
        self.btn_folder.setFixedHeight(46)
        self.btn_folder.setFixedWidth(130)
        self.btn_folder.setFont(QFont("Segoe UI", 12))
        
        row6.addWidget(self.btn_folder)
        row6.addStretch()
        row6.addWidget(self.btn_start)
        row6.addWidget(self.btn_skip_break)
        row6.addWidget(self.btn_save)


        # Add created layers to layout ---->
        self.lay.addLayout(row1)
        self.lay.addWidget(self.pgbar_total)
        self.lay.addWidget(instr_box, stretch=3)
        self.lay.addWidget(self.pgbar_trial)
        self.lay.addLayout(row5)
        self.lay.addLayout(row6)
 

    def _update_class_box(self, cls: str, count: int):
        pct = count / self.TRIALS_PER_CLASS
        color = self.COLORS[cls]
        bg = self.COLORS["surface"]

        if pct >= 1.0:
            boxStyle = f"""
                QWidget {{
                    background: {color};
                    border-radius: 10px;
                }}
                QLabel {{ 
                    background: transparent; color: #e0e0e0; 
                }}
            """
        elif pct <= 0.0:
            boxStyle = f"""
                QWidget {{
                    background: {self.COLORS['surface']};
                    border-radius: 10px;
                }}
                QLabel {{ 
                    background: transparent; color: #e0e0e0; 
                }}
            """
        else:
            boxStyle = f"""
                QWidget {{
                    background: qlineargradient(
                        x1:0, y1:0, x2:1, y2:0,
                        stop:0          {color},
                        stop:{pct:.3f}  {color},
                        stop:{pct + 0.001:.3f} {bg},
                        stop:1          {bg}
                    );
                    border-radius: 10px;
                }}
                QLabel {{ 
                    background: transparent; color: #e0e0e0; 
                }}
            """

        self.class_boxes[cls].setStyleSheet(boxStyle)
  

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.COLORS["bg"]};
                color: {self.COLORS["text"]};
            }}

            #instrBox {{
                background-color: {self.COLORS["surface"]};
                border-radius: 14px;
            }}

            #instrBox QLabel {{
                background: transparent;
            }}

            #classBox {{
                background: {self.COLORS['surface']};
                border-radius: 10px;
            }}

            #classBox QLabel {{
                background: transparent;
                color: {self.COLORS['text']};
            }}

            QPushButton {{
                background-color: #383838;
                color: {self.COLORS["text"]};
                border: 1px solid #555;
                border-radius: 8px;
                padding: 0 16px;
            }}

            QPushButton:hover {{ 
                background-color: #484848; 
            }}

            QPushButton:disabled {{ 
                color: #555; 
                border-color: #444; 
            }}

            QProgressBar {{
                background-color: #383838;
                border-radius: 4px;
                border: none;
            }}

            QProgressBar::chunk {{
                background-color: #4169E1;
                border-radius: 4px;
            }}
        """)


    # FUNCTION GROUP 2: SESSION CONTROL
    def _make_trial_order(self) -> list:
        order = []

        for _ in range(self.N_BLOCKS):
            block = []
            for i in range(self.N_CLASSES):
                block.extend([i] * self.TRIALS_PER_BLOCK)
            np.random.shuffle(block)
            order.extend(block)

        return order
    

    def _start_session(self):
        self.collected_eeg = []
        self.collected_labels = []
        self.current_trial = 0
        self.trial_order = self._make_trial_order()

        self.btn_start.setEnabled(False)
        self.btn_save.setEnabled(True)
        self._next_trial()


    def _stop_and_save(self):
        self._eeg_worker.stop_recording() 
        self._eeg_worker.stop_simulation()
        self.timer.stop()
        self.btn_save.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_skip_break.setEnabled(False)

        if not self.collected_eeg:
            self.lb_cue.setText("No data to save")
            return
    
        import os 
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        eeg_arr = np.stack(self.collected_eeg, axis=0).astype(np.float32)
        labels_arr = np.array(
            [self.CLASSES.index(l) for l in self.collected_labels],
            dtype=np.int32
        )

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(self.SAVE_PATH, f"data_session_{ts}.npz")

        np.savez(
            file_name,
             eeg_data = eeg_arr,
             labels = labels_arr,
             class_names = np.array(self.CLASSES)
        )

        from collections import Counter
        counts = Counter(labels_arr.tolist())
        summary = " | ".join(f"{self.CLASSES[k]}: {v}" for k, v in sorted(counts.items()))
        logger.info(f"Saved {file_name} - {len(labels_arr)} trials")

        self.lb_cue.setText(f"{len(labels_arr)} trials saved ✓")
        self.lb_cue.setStyleSheet("color: #2E8B57;")
        self.lb_phase.setText(os.path.basename(file_name))

    # FUNCTION GROUP 3: TRIAL FLOW

    def _next_trial(self):
        total_trials = len(self.trial_order)
        block_size = self.N_CLASSES * self.TRIALS_PER_BLOCK

        if self.current_trial >= total_trials:
            self._finish()
            return
        
        current_block = self.current_trial // block_size + 1
        
        # Check for new block
        if not self._in_break:
            if self.current_trial > 0 and self.current_trial % block_size == 0:
                current_block -= 1
                self._start_break(current_block)
                return
        
        self._in_break = False

        # Update UI header
        self.lb_block.setText(f"Block {current_block} / {self.N_BLOCKS}")
        self.lb_trial.setText(f"Trial {self.current_trial + 1} / {total_trials}")
        self.pgbar_total.setValue(self.current_trial)

        # Get current task-type
        current_task_no = self.trial_order[self.current_trial]
        current_task_type = self.CLASSES[current_task_no]
        self._current_task_type = current_task_type
        arrows  = {
            "Left Hand": "←", 
            "Rest": "●", 
            "Right Hand": "→"
        }

        # Update instruction box
        self.lb_cue.setText(current_task_type)
        self.lb_cue.setStyleSheet(f"color: {self.COLORS[current_task_type]};")
        self.lb_arrow.setText(arrows[current_task_type])
        self.lb_arrow.setStyleSheet(f"color: {self.COLORS[current_task_type]};")
        self.lb_phase.setText("Get ready...")

        self._set_phase("prepare", self.PREPARE_SECONDS)


    def _set_phase(self, phase: str, duration: float):
        self.phase = phase
        self.phase_start = time.time()
        self.phase_duration = duration
        self.pgbar_trial.setValue(0)
        self.timer.start(50)   # tick each 50ms


    def _tick(self):
        elapsed = time.time() - self.phase_start
        pct = int(min(elapsed / self.phase_duration * 100, 100))
        self.pgbar_trial.setValue(pct)

        if elapsed < self.phase_duration:
            return
        
        self.timer.stop()

        if self.phase == "prepare":
            self.lb_phase.setText("⬤  RECORDING")
            self._eeg_worker.start_recording(self._current_task_type)
            logger.info(">>> start_recording called")
            self._set_phase("record", self.TRIAL_SECONDS)

        elif self.phase == "record":
            self._eeg_worker.stop_recording()
            logger.info(">>> stop_recording called")
            self.lb_cue.setText("Rest")
            self.lb_cue.setStyleSheet(f"color: {self.COLORS['text']};")
            self.lb_arrow.setText("")
            self.lb_phase.setText("")
            self._set_phase("rest", self.REST_SECONDS)

        elif self.phase == "rest":
            self.current_trial += 1
            self._next_trial()

        elif self.phase == "break":
            self.btn_skip_break.setEnabled(False)
            self._next_trial()


    # FUNCTION GROUP 4: BREAK

    def _start_break(self, completed_block: int):
        self._in_break = True
        self.lb_block.setText(f"Block {completed_block} / {self.N_BLOCKS}")
        self.lb_cue.setText(f"Block {completed_block} done!")
        self.lb_cue.setStyleSheet("color: #2E8B57;")
        self.lb_arrow.setText("☕")
        self.lb_arrow.setStyleSheet("color: #e0e0e0;")
        self.lb_phase.setText(f"Break — next block in {self.BREAK_SECONDS}s")
        self.btn_skip_break.setEnabled(True)
        self._set_phase("break", self.BREAK_SECONDS)


    def _skip_break(self):
        self.timer.stop()
        self.btn_skip_break.setEnabled(False)
        self._next_trial()


    # FUNCTION GROUP 5: DATA

    def _on_trial_ready(self, eeg_array: np.ndarray, label: str):
        if not label:
            logger.warning("on_trial_ready: empty label, ignoring")
            return

        self.collected_eeg.append(eeg_array)
        self.collected_labels.append(label)

        count = self.collected_labels.count(label)
        self.count_labels[label].setText(f"{count} / {self.TRIALS_PER_CLASS}")
        self._update_class_box(label, count)

        logger.info(f"Trial {len(self.collected_labels)} recorded: {label}")


    # FUNCTION GROUP 6: OTHERS

    def _open_folder(self):
        import subprocess, platform, os
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        if platform.system() == "Windows":
            os.startfile(self.SAVE_PATH)

        elif platform.system() == "Darwin":
            subprocess.Popen(["open", self.SAVE_PATH])

        else:
            subprocess.Popen(["xdg-open", self.SAVE_PATH])


    def _finish(self):
        self.timer.stop()
        self.lb_cue.setText("Session complete!")
        self.lb_cue.setStyleSheet("color: #2E8B57;")
        self.lb_arrow.setText("✓")
        self.lb_arrow.setStyleSheet("color: #2E8B57;")
        self.lb_phase.setText("Press STOP & SAVE to save data")
        self.pgbar_total.setValue(self.pgbar_total.maximum())
        self.btn_skip_break.setEnabled(False)

    

if __name__ == "__main__":
    import json
    from pathlib import Path
    from bcg_core.config_schema import DataCollectConfig

    with open("config/datacollect_conf.json") as f:
        config = DataCollectConfig.model_validate(json.load(f))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    data_collect_window = DataCollectionWindow(config)
    data_collect_window.show()

    sys.exit(app.exec())
