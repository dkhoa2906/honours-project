"""WebSocket sequence with a fake browser: session_start, trial_start/trial_end, save."""
import asyncio
import json
import socket
import threading
import time

import numpy as np
import pytest
import websockets

from bcg_server.bcg_server import BCGServer
from conftest import server_cfg

GAME_KEYS = {"eeg_data", "labels", "class_names", "method", "sampling_rate", "trial_seconds"}
WINDOW_S = 0.6            # real time per trial window; trial_seconds=0.5 -> 64 samples, kept if >= 51


def free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(tmp_path):
    port = free_port()
    srv = BCGServer(server_cfg(tmp_path, trial_seconds=0.5, port=port), signals=None)
    threading.Thread(target=lambda: asyncio.run(srv.run()), daemon=True).start()
    stop = threading.Event()

    def feed():                                        # float32 arrays, like CortexReader
        k, nxt = 0, time.perf_counter()
        while not stop.is_set():
            srv._on_sample(np.full(14, k % 1000, dtype=np.float32)); k += 1
            nxt += 1 / 128
            time.sleep(max(0.0, nxt - time.perf_counter()))

    for _ in range(100):                               # wait for the listener
        try:
            socket.create_connection(("localhost", port), timeout=0.1).close(); break
        except OSError:
            time.sleep(0.05)
    threading.Thread(target=feed, daemon=True).start()
    yield srv, port, tmp_path
    stop.set()


class Browser:
    """Minimal fake of game/js: sends messages and records everything the server returns."""

    def __init__(self, ws):
        self.ws, self.got = ws, []
        self.task = asyncio.ensure_future(self._read())

    async def _read(self):
        try:
            async for raw in self.ws:
                self.got.append(json.loads(raw))
        except websockets.ConnectionClosed:
            pass

    async def send(self, **msg):
        await self.ws.send(json.dumps(msg))

    async def trial(self, label):
        await self.send(type="trial_start", label=label)
        await asyncio.sleep(WINDOW_S)
        await self.send(type="trial_end", label=label)
        await asyncio.sleep(0.15)

    def of_type(self, t):
        return [m for m in self.got if m["type"] == t]


def run(coro):
    return asyncio.run(coro)


def test_full_sequence_and_save(live_server):
    srv, port, out = live_server

    async def scenario():
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            b = Browser(ws)
            await asyncio.sleep(0.3)
            await b.send(type="session_start")
            for lb in ["Left Hand", "Rest", "Right Hand", "Rest"]:
                await b.trial(lb)
            await b.send(type="save_game_session", min_trials=4)
            await asyncio.sleep(0.4)
            return b.got

    got = run(scenario())
    types = [m["type"] for m in got]
    assert types[0] == "status"
    assert [m["count"] for m in got if m["type"] == "trial_count"] == [0, 1, 2, 3, 4]
    saved = [m for m in got if m["type"] == "session_saved"][0]
    assert saved["trials"] == 4 and saved["complete"] is True
    d = np.load(saved["path"], allow_pickle=False)
    assert set(d.files) == GAME_KEYS
    assert d["labels"].tolist() == [0, 1, 2, 1] and d["eeg_data"].shape == (4, 64, 14)
    assert srv._trials == []                                # cleared after the save


def test_eeg_window_is_valid_json_with_float32_samples(live_server):
    srv, port, _ = live_server

    async def scenario():
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            b = Browser(ws)
            await asyncio.sleep(1.2)
            return b.of_type("eeg_window")

    windows = run(scenario())
    assert windows and len(windows[0]["data"]) == 64 and len(windows[0]["data"][0]) == 14
    assert isinstance(windows[0]["data"][0][0], float)


def test_incomplete_save_when_pool_runs_out(live_server):
    srv, port, out = live_server

    async def scenario():
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            b = Browser(ws)
            await b.send(type="session_start")
            await b.trial("Left Hand"); await b.trial("Rest")
            await b.send(type="save_game_session", min_trials=6)                      # refused
            await asyncio.sleep(0.3)
            assert b.of_type("error") and not b.of_type("session_saved")
            await b.send(type="save_game_session", min_trials=6, allow_incomplete=True)
            await asyncio.sleep(0.4)
            return b.of_type("session_saved")[0]

    saved = run(scenario())
    assert saved["complete"] is False and saved["trials"] == 2 and saved["required"] == 6
    d = np.load(saved["path"], allow_pickle=False)
    assert set(d.files) == GAME_KEYS | {"complete", "n_trials"}
    assert d["n_trials"].item() == 2 and not d["complete"].item()


def test_reconnect_keeps_trials_and_session_start_clears(live_server):
    srv, port, _ = live_server

    async def scenario():
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            b = Browser(ws)
            await b.send(type="session_start")
            await b.trial("Left Hand")
        await asyncio.sleep(0.3)
        async with websockets.connect(f"ws://localhost:{port}") as ws2:       # reconnect, same session
            b2 = Browser(ws2)
            await asyncio.sleep(0.2)
            kept = len(srv._trials)
            await b2.trial("Right Hand")
            n_after = len(srv._trials)
            await b2.send(type="session_start")                                # Start pressed again
            await asyncio.sleep(0.3)
            return kept, n_after, len(srv._trials)

    kept, n_after, after_start = run(scenario())
    assert (kept, n_after, after_start) == (1, 2, 0)


def test_second_client_is_refused_and_does_not_touch_data(live_server):
    srv, port, _ = live_server

    async def scenario():
        async with websockets.connect(f"ws://localhost:{port}") as ws:
            b = Browser(ws)
            await b.send(type="session_start")
            await b.trial("Rest")
            async with websockets.connect(f"ws://localhost:{port}") as ws2:
                first = json.loads(await ws2.recv())
            return first, len(srv._trials)

    first, n = run(scenario())
    assert first["type"] == "error" and n == 1
