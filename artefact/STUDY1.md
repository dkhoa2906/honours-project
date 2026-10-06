# Study 1: exact configuration

Study 1 is the pilot user study described in the paper. EEG was **simulated** (no real EEG was recorded) and the study measured perceived workload (NASA-TLX). Participant details are in the paper, not in this repository.

## Code

- Git tag `study1` = commit `750cd3e`. Its artefact code is identical to `9332380` (2026-06-11), the last commit that changed it.
- To run exactly that code: `git checkout study1`.
- The current `cleanup` branch reproduces the same protocol with its default settings (see "Preset" below). `CHANGELOG.md` lists everything that differs.

## Simulated EEG

`simulation_mode: true` makes `EEGWorker` emit 14 channels of the constant value 36.0 at 128 Hz (`bcg_core/eeg_worker.py`, `_generate_fake_sample`). Saved Study 1 style files therefore have `min = max = 36.0`; they show the file layout, not signal quality.

## Graz protocol (`config/datacollect_conf.example.json`)

| Setting | Value |
|---|---|
| `preprocessing.sampling_rate` | 128 Hz |
| `recording.prepare_seconds` | 1.0 s (cue already shown) |
| `recording.trial_seconds` | 4.0 s (recorded, 512 samples) |
| `recording.rest_seconds` | 2.5 s (not recorded) |
| `recording.n_blocks`, `trials_per_block` | 1 block, 26 trials per class |
| `recording.break_seconds` | 0 (no break with one block) |
| `recording.labels` | Left Hand, Rest, Right Hand |
| Order | each class 26 times, shuffled with `np.random.shuffle` |

Per trial 1.0 + 4.0 + 2.5 = 7.5 s; 78 trials = 585 s (9.75 min). Phase times are checked by a 50 ms `QTimer`.

## Game (`game/js/config.js`, `game/js/study.js`, `config/server_conf.example.json`)

| Setting | Value |
|---|---|
| Lanes | 2 (LEFT = Left Hand, RIGHT = Right Hand) |
| Pool | 26 tiles per lane, shuffled with `Math.random()` (52 tiles) |
| First tile | 2000 ms after Start |
| Tile fall to the hit line | 440 px at 3 px per frame = 147 frames, about 2.45 s on a 60 Hz display |
| Motor epoch | `TRIAL_DURATION_MS` = 4000 ms, starts when the tile reaches the hit line |
| Pause | `REST_DEADZONE_MS` = 500 ms |
| Rest epoch | `REST_DURATION_MS` = 4000 ms |
| Gap before the next tile | `INTER_TRIAL_MS` = 500 ms |
| Server epoch | `server.trial_seconds` = 4.0 s, 512 samples, kept if at least 80 % |
| Stream step | `live.step_samples` = 64 |
| Port | `ws://localhost:8765` |

Each tile gives one motor trial and one Rest trial, so one tile cycle is 4.0 + 0.5 + 4.0 + 0.5 = 9.0 s plus the fall time.

## Preset: labeled trials per session

The number of labeled trials is set in `game/js/study.js` (`DEFAULT_PRESET`) or per page load with `index.html?preset=<name>`.

| Preset | Labeled trials | Content | Duration (60 Hz, no rejected trials) |
|---|---|---|---|
| `study1` (default) | 78 | 39 motor + 39 Rest. Only the first 39 tiles of the shuffled pool are played, so left and right counts are not forced to be equal (the sample game recording has 20 and 19). | 39 x (9.0 + 2.45) s + 2 s = about 449 s, 7.5 min |
| `full_pool` | 104 | 52 motor + 52 Rest: the whole pool, 26 tiles per lane | 52 x (9.0 + 2.45) s + 2 s = about 597 s, 9.9 min |

The session is saved when the server has accepted that many trials. If the tile pool runs out first (trials were rejected), the game asks the server to save what exists; that file gets `complete = False` and `n_trials`, which normal files do not have.

## Reproduce

```bash
cp config/datacollect_conf.example.json config/datacollect_conf.json
cp config/server_conf.example.json      config/server_conf.json
python main.py
# Graz: Data Collect Tool -> START ... STOP & SAVE
# Game: BCG Game Server -> Start Server; open game/index.html, Connect, Start
```

Files are written to `recordings/` as `graz_session_<timestamp>.npz` and `bcg_game_session_<timestamp>.npz`.
