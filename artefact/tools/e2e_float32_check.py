"""Run: python tools/e2e_float32_check.py [output_dir]

End-to-end check of BCGServer with ndarray samples (as CortexReader emits: np.float32 arrays).
Sample k carries value k on all 14 channels, so alignment of each saved trial can be verified.
Uses trial_seconds=1.0 (128 samples) to keep run short. Server code is NOT modified."""
import asyncio, json, sys, tempfile, threading, time
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, websockets
from bcg_server import BCGServer
PORT, OUT = 8792, (sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='bcg_e2e_'))
cfg = {"preprocessing": {"sampling_rate": 128}, "model": {"n_channels": 14},
       "live": {"step_samples": 64, "simulation_mode": True, "classes": ["Left Hand", "Rest", "Right Hand"],
                "confidence_threshold": 0.5, "model_path": "x"},
       "server": {"host": "localhost", "port": PORT, "trial_seconds": 1.0, "calibration_epochs": 1, "save_dir": OUT}}
srv = BCGServer(cfg, signals=None)
send_errors = []
_orig = srv.send
async def _wrapped(msg):
    try: await _orig(msg)
    except Exception as e: send_errors.append((msg.get("type"), type(e).__name__, str(e)[:80])); raise
srv.send = _wrapped
threading.Thread(target=lambda: asyncio.run(srv.run()), daemon=True).start(); time.sleep(1)
stop = threading.Event(); k = [0]
def feed():
    nxt = time.perf_counter()
    while not stop.is_set():
        srv._on_sample(np.full(14, k[0], dtype=np.float32)); k[0] += 1
        nxt += 1/128; d = nxt - time.perf_counter()
        if d > 0: time.sleep(d)
async def main():
    got = {}
    async with websockets.connect(f"ws://localhost:{PORT}") as ws:
        async def reader():
            async for raw in ws:
                m = json.loads(raw); got[m["type"]] = got.get(m["type"], 0) + 1
                if m["type"] in ("error", "session_saved"): print("server msg:", m)
        rt = asyncio.create_task(reader())
        threading.Thread(target=feed, daemon=True).start(); await asyncio.sleep(1)
        labels = ["Left Hand", "Rest", "Right Hand"] * 10
        for lb in labels:
            await ws.send(json.dumps({"type": "trial_start", "label": lb})); await asyncio.sleep(1.0)
            await ws.send(json.dumps({"type": "trial_end", "label": lb}));   await asyncio.sleep(0.2)
        await ws.send(json.dumps({"type": "save_game_session", "min_trials": 30})); await asyncio.sleep(1)
        rt.cancel()
    stop.set()
    print("messages received by client:", got)
    print("send() exceptions inside server:", {e[:2] for e in send_errors}, "count", len(send_errors), "| sample:", send_errors[:1])
asyncio.run(main())
import glob
f = sorted(glob.glob(OUT + "/bcg_game_session_*.npz"))[-1]; d = np.load(f, allow_pickle=False)
X = d["eeg_data"]; print("saved:", f.split("/")[-1], {k: (d[k].dtype, d[k].shape) for k in d.files})
bad = 0
for i in range(len(X)):
    v = X[i, :, 0]; steps = np.unique(np.diff(v)); ok = len(steps) == 1 and steps[0] == 1
    if not ok: bad += 1; print("  trial", i, "non-consecutive steps:", steps[:5], "first/last", v[0], v[-1])
print(f"trials={len(X)} non-consecutive={bad}; first-sample index of trials:", X[:6, 0, 0].astype(int).tolist())
