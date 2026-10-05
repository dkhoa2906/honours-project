# SPDX-License-Identifier: GPL-3.0-or-later
"""WebSocket ping -> pong round-trip time against the real BCGServer class (headless).

This is the latency measurement reported in the paper: local machine, Python client,
n = 200 timed pings (after 20 warm-ups). The server answers {"type":"ping"} with
{"type":"pong"}. Condition A: idle server. Condition B: the server also receives a
simulated 128 Hz EEG feed.

Run from anywhere:  python tools/ws_rtt.py [output_dir]
Needs `websockets` (requirements.txt). Writes nothing to recordings/.
"""
import asyncio, json, statistics, sys, tempfile, threading, time
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import websockets
from bcg_server import BCGServer

PORT = 8791
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='bcg_rtt_')
cfg = {"preprocessing": {"sampling_rate": 128}, "model": {"n_channels": 14},
       "live": {"step_samples": 64, "simulation_mode": True, "classes": ["Left Hand", "Rest", "Right Hand"],
                "confidence_threshold": 0.5, "model_path": "x"},
       "server": {"host": "localhost", "port": PORT, "trial_seconds": 4.0, "calibration_epochs": 1, "save_dir": SCRATCH}}
srv = BCGServer(cfg, signals=None)
threading.Thread(target=lambda: asyncio.run(srv.run()), daemon=True).start()
time.sleep(1.0)

feed_on = threading.Event(); stop = threading.Event()
def feed():
    period = 1/128; nxt = time.perf_counter()
    while not stop.is_set():
        if feed_on.is_set(): srv._on_sample([36.0]*14)
        nxt += period; d = nxt - time.perf_counter()
        if d > 0: time.sleep(d)
threading.Thread(target=feed, daemon=True).start()

async def run(n, label):
    rtts = []
    async with websockets.connect(f"ws://localhost:{PORT}") as ws:
        await ws.recv()                      # initial {"type":"status"}
        for _ in range(20):                  # warm-up (discarded)
            await ws.send('{"type":"ping"}')
            while json.loads(await ws.recv()).get("type") != "pong": pass
        for _ in range(n):
            t0 = time.perf_counter_ns()
            await ws.send('{"type":"ping"}')
            while json.loads(await ws.recv()).get("type") != "pong": pass   # skip interleaved eeg_window msgs
            rtts.append((time.perf_counter_ns() - t0) / 1e6)
            await asyncio.sleep(0.01)
    s = sorted(rtts)
    print(f"{label}: n={len(rtts)} mean={statistics.mean(rtts):.3f} ms  SD={statistics.stdev(rtts):.3f} ms  "
          f"min={s[0]:.3f}  median={statistics.median(rtts):.3f}  p95={s[int(.95*len(s))-1]:.3f}  max={s[-1]:.3f} ms")

async def main():
    await run(200, "A idle server            ")
    feed_on.set(); await asyncio.sleep(0.5)
    await run(200, "B + 128 Hz simulated feed")
asyncio.run(main()); stop.set()
