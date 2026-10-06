# SPDX-License-Identifier: GPL-3.0-or-later
# Which thread runs a plain-Python-method slot when the signal is emitted from a non-Qt thread?
# Mirrors: EEGWorker(QObject).sample_ready (pyqtSignal(object)) -> BCGServer._on_sample (non-QObject method)
import threading, sys
from PyQt6.QtCore import QCoreApplication, QObject, pyqtSignal, QTimer
class W(QObject): sig = pyqtSignal(object)
class Plain:
    def slot(self, x): print("slot ran on:", threading.current_thread().name)
app = QCoreApplication(sys.argv); w = W(); p = Plain(); w.sig.connect(p.slot)
print("GUI/main thread:", threading.current_thread().name)
t = threading.Thread(target=lambda: (print("emitting from:", threading.current_thread().name), w.sig.emit([1.0])), name="cortex-ws-thread"); t.start(); t.join()
QTimer.singleShot(300, app.quit); app.exec()
