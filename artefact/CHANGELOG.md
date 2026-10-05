# Changelog

Branch `cleanup`, relative to tag `study1` (commit `750cd3e`).

**None of the items that the paper describes was changed**: 512 samples at 128 Hz and class codes 0 Left, 1 Rest, 2 Right; the 80 % window check with padding by the last sample and trimming (Graz keeps the first 512, the game server the last 512); the Graz timeline (1.0 / 4.0 / 2.5 s, 26 trials per class, one block, random order, 50 ms timer); the game timeline and tile pool (two lanes, 4.0 / 0.5 / 4.0 / 0.5 s, 26 tiles per lane, trial boundaries from the server-side sample counter); the 78 trials of Study 1 (still the default); the NPZ keys of normal sessions; no filtering and no artifact rejection at collection; no keyboard, mouse or touch handlers in the game. Tests in `tests/` check the window rules and the NPZ keys; `git diff study1 HEAD -- game/js/config.js` shows only the license line.

## Behavior changes

### Data
- **Cortex electrode columns** (`bcg_core/cortex_reader.py`). The code took `eeg[1:15]`. Electrodes are now selected by name from the `cols` of the subscribe response (fallback `[2:16]`, the chosen path is logged). With the documented packet layout the old slice made channel 0 the INTERPOLATED flag and dropped AF4. Only fake packets built from the Cortex documentation were used to test this; it has not been run with a headset. The shape stays (n, 512, 14). Subscribe results that list streams as dicts are now accepted.
- **Class order** (`bcg_core/config_schema.py`). One definition, `CLASS_ORDER`. Calibration trained with Left/Right/Rest = 0/1/2 and now uses 0/1/2 = Left/Rest/Right like the saved files and the classifier. The Graz `labels` setting must equal this order. The game server always saves `class_names` in this order (it no longer reads `live.classes` for this).
- **Incomplete game sessions.** If the tile pool runs out before the target trial count, the session is saved with two extra keys, `complete` (False) and `n_trials`. Normal sessions are unchanged.
- **Relative paths** in configs (recordings folder, model weights) resolve against the repository root instead of the current directory or `bcg_server/`. Calibration now finds `models/eegnet_finetuned_mimed.pth`; before, it trained from random weights without saying so.

### Server and game
- New message `session_start` (sent by the game when Start is pressed). The server clears its trial list on it and after a successful save. It does not clear on connect. A second click on Start while running is ignored.
- The game's labeled-trial count lives in `game/js/study.js` (presets `study1` = 78 default, `full_pool` = 104; `?preset=` in the page address).
- Every rejected trial is logged with its reason (too short / wrong shape) and a wrong-shape rejection now also sends an `error` message to the browser.
- `eeg_window` messages no longer fail with float32 samples (they raised `TypeError` and were never sent).
- The sample counter, buffer and stream counter are guarded by one lock; the Graz trial buffer likewise. Timing is not changed.
- `session_saved` carries `complete` and `required`; the game's info panel shows an incomplete save.

### Graz
- After STOP & SAVE the sample source is restarted on the next START (it was shut down while START was re-enabled).

### Classifier (experimental)
- `RealtimeClassifier.predict` returns `(None, 0.0)` when it fails instead of `("Rest", 0.0)`. The live monitor shows "No prediction" and the server sends an `error` message instead of a prediction.

## Repository
- Added: `README.md`, `STUDY1.md`, `THIRD_PARTY.md`, `LICENSE` (GPL-3.0-or-later), SPDX line in every source file, `tests/` (pytest), `tools/` (including `ws_rtt.py`, the latency measurement), `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `bcg_core/paths.py`, `bcg_core/log.py`.
- `config/*_template.json` became `config/*.example.json` (a server example was added; the Graz example has the Study 1 values). Real `config/*.json` stay ignored.
- Model weights (`*.pth`) are ignored and no longer tracked; earlier commits still contain them.
- Removed unused imports and attributes; one logging setup instead of three `basicConfig` calls; docstrings on public classes and functions.
