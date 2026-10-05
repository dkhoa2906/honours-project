# BCG: Brain-Computer Game

A dual-mode framework for collecting labeled motor imagery (MI) EEG data. It supports the traditional **Graz protocol** and a **gamified protocol** in which a two-lane rhythm game provides the cue. It accompanies the paper *BCG: A Game-Based Approach to Motor Imagery Adaptation and BCI Calibration*.

> **Status.** BCG is a data-collection platform, not a classification system. It has **not been validated on real EEG recordings**. The pilot user study that goes with the paper measured perceived workload only; its EEG stream was simulated.

## What it does

- Records labeled 4.0 s EEG epochs (512 samples at 128 Hz) for three classes: Left Hand, Right Hand, Rest.
- Graz protocol: a desktop application shows cues and records timed windows.
- Gamified protocol: a browser game sends trial start and end events to a local server; the lane of each tile gives the intended class, so no manual annotation is needed.
- Saves each session as one `.npz` file for offline analysis.
- Works with an Emotiv EPOC X (14 electrodes) through the Cortex API, or with a built-in simulation mode that needs no headset.

## How it works

```
Acquisition   EPOC X -> Cortex API -> CortexReader -> EEGWorker
Collection    Graz collect tool (desktop)   |   Game server <-> browser game (WebSocket)
Storage       session-level .npz files
```

Both protocols share the acquisition path. They differ in how a trial is cut from the stream:

- **Graz:** a software timer (checked every 50 ms) starts and stops recording.
- **Game:** the server counts incoming EEG samples (one count per sample) and notes the count when the browser's start and end messages arrive. The trial is the set of samples between the two counts.

## Protocols (as used in the user study)

| | Graz | Gamified |
|---|---|---|
| Preparation | 1.0 s | tile fall, about 2.45 s (60 Hz display) |
| Recorded epoch | 4.0 s | 4.0 s |
| Rest epoch | 4.0 s, cued as a class | 4.0 s, starts 0.5 s after the motor epoch |
| Pause between trials | 2.5 s (not recorded) | 0.5 s |
| Labeled trials | 78 (26 per class) | 78 (about 39 motor and 39 Rest) |
| Order | random, single block | random lane order, Rest after each motor trial |

Labels come from the protocol (the cued class, or the lane), not from the EEG. They denote the **intended** class and do not confirm that imagery was performed.

## Quick start

Requirements: Python 3.13 (developed and run on macOS with 3.13.2; other systems and versions are untested), the packages in `requirements.txt` (PyQt6, numpy, scipy, torch, braindecode, pydantic, websocket-client, websockets), and a browser for the game.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config/datacollect_conf.example.json config/datacollect_conf.json
cp config/milive_conf.example.json      config/milive_conf.json
cp config/server_conf.example.json      config/server_conf.json
```

The example configs have `simulation_mode: true` and the Study 1 values for the Graz tool. Model weights (`*.pth`) are not in the repository; the live monitor and the calibration code need `models/eegnet_finetuned_mimed.pth`, which you place there yourself. The collection tools do not need it.

Run in **simulation mode** (no headset; emits fixed-value 14-channel samples at 128 Hz):

```bash
python main.py    # launcher: "Data Collect Tool" (Graz), "Live MI Monitor", "BCG Game Server"
```

- Graz: click **Data Collect Tool**, then START. Press STOP & SAVE to write the file.
- Game: click **BCG Game Server**, then **Start Server** (default address `localhost:8765`). Open `game/index.html` in a browser, click **Connect**, then **Start**. The repo has no web server for the game page. Add `?preset=full_pool` to the page address for the longer session (see below).

With a real headset: start Emotiv Launcher (Cortex listens on `wss://localhost:6868`), grant the app permission, set `"simulation_mode": false` in the config and put your Cortex `client_id` and `client_secret` under `cortex_api`. The real configs `config/*.json` are gitignored. Only the `*.example.json` files are tracked, with empty credentials. Never commit credentials.

## Output format

One `.npz` per session in `recordings/` (Graz: `recording.save_path`; game: `server.save_dir`, default `recordings/`), named `graz_session_YYYYMMDD_HHMMSS.npz` or `bcg_game_session_YYYYMMDD_HHMMSS.npz`:

| Key | Type and shape | Content | Written by |
|---|---|---|---|
| `eeg_data` | float32, (trials, 512, 14) | trial x time x channel, as delivered by Cortex, unfiltered | both |
| `labels` | int32, (trials,) | 0 Left Hand, 1 Rest, 2 Right Hand | both |
| `class_names` | string, (3,) | Left Hand, Rest, Right Hand | both |
| `sampling_rate` | scalar | 128 (Hz) | game only |
| `trial_seconds` | scalar | 4.0 | game only |
| `method` | string | "game" | game only |

Not stored: subject or session IDs, per-trial timestamps, channel names. Example:

```python
import numpy as np
d = np.load("recordings/graz_session_20260609_154542.npz")
X, y, names = d["eeg_data"], d["labels"], d["class_names"]
print(X.shape, X.dtype, names[y[0]])   # (38, 512, 14) float32 Rest
```

## Window checks, preprocessing and artifacts

- A window is kept if it has at least about 80% of 512 samples; shorter windows are discarded. Short windows are padded by repeating the last sample; long windows are trimmed (Graz keeps the first 512 samples, the game server the last 512). Lost samples are not detected.
- Saved epochs are **not filtered by BCG**. The headset applies its own filtering (0.2-45 Hz, notch at 50 and 60 Hz). An offline preprocessing step (5th-order Butterworth band-pass 8-30 Hz, 50 Hz notch, z-scoring per channel, zero-phase) is part of the experimental classification tools. It is `EEGPreprocessor.process` in `bcg_core/classifier.py`.
- There is **no automatic artifact rejection**. Blinks, muscle activity and electrode movement stay in the saved epochs.

## Known limitations

- The EPOC X has no electrodes at C3, Cz, C4, so MI-related activity may be captured only indirectly.
- Trial boundaries in the game have an unquantified timing error. Only the local WebSocket round trip was measured (0.9 ms on average, local machine, Python client).
- Game and Graz sessions differ in length and in the number of motor trials (see the table above).
- The Live MI Monitor and the classifier code are experimental and not part of the collection workflow.
- Open issues: electrode columns are selected by name from the Cortex subscribe response, but this was tested only with fake packets built from the Cortex documentation, never with a headset; the Graz tool counts only trials that pass the 80 % rule and does not show how many were dropped.

## Reproducing the paper's setup

- The Study 1 configuration is the default (preset `study1` in `game/js/study.js`): 78 labeled trials per session. A second preset plays the game's full tile pool (52 motor and 52 Rest trials, about 10 minutes). Git tag `study1` marks the code used for the study.
- Exact settings and timings: `STUDY1.md`. What changed since the study: `CHANGELOG.md`. Tests: `pip install -r requirements-dev.txt && pytest`.

## Repository layout

```
main.py           launcher (Graz tool, live monitor, game server)
bcg_core/         Cortex reader, EEGWorker, classifier and preprocessing, config schema
bcg_collect/      Graz protocol window
bcg_server/       WebSocket game server and its window
bcg_live/         Live MI Monitor (experimental)
game/             browser game (index.html, js/, css/)
config/           *.example.json templates; real configs are gitignored
models/           place EEGNet weights here (not tracked)
recordings/       saved sessions (gitignored)
tests/, tools/    pytest suite; latency, thread and file-inspection scripts
```

## Acknowledgements

Brain-Life Link Technology JSC lent the EEG headset. Emotiv sponsored the software license.

## License

BCG is released under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See `LICENSE` and `THIRD_PARTY.md`. Third-party logos and trademarks (for example Emotiv) remain the property of their owners and are not covered by this license.
