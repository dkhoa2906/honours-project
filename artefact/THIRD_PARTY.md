# Third-party material

BCG's own code is GPL-3.0-or-later (see `LICENSE`). This file lists what else the repository uses or ships, with the license as stated in the installed package metadata or found in the repository. Where I could not determine a license, it says so. Nothing here was deleted or replaced.

Third-party logos and trademarks (for example Emotiv, Birmingham City University, UIT) are not covered by the project license.

## Python dependencies the BCG code imports

| Package (pinned version) | License (installed metadata) | GPLv3 compatible |
|---|---|---|
| PyQt6 6.11.0 | GPL-3.0-only (Riverbank also sells a commercial license) | Yes. The combined application is distributed under GPLv3 terms; "or later" cannot be used for the combination. |
| PyQt6-Qt6 6.11.0 (Qt libraries) | LGPL v3 | Yes |
| PyQt6_sip 13.11.1 | BSD-2-Clause | Yes |
| numpy 2.4.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Yes |
| scipy 1.17.1 | metadata gives the long BSD-style notice text (SciPy is BSD-3-Clause) | Yes |
| torch 2.11.0 | BSD-3-Clause | Yes |
| braindecode 1.4.0 | BSD-3-Clause | Yes |
| pydantic 2.13.1 / pydantic_core 2.46.1 | MIT | Yes |
| websocket-client 1.9.0 | Apache-2.0 | Yes (GPLv3, not GPLv2) |
| websockets 17.2 | BSD-3-Clause | Yes |

## Other pinned packages

`requirements.txt` is a freeze of the development venv and also holds packages used only by the notebooks in `../model` and transitive dependencies. Licenses from installed metadata:

- MIT: annotated-types, attrs, cffi, charset-normalizer, einops, filelock, fonttools, platformdirs, pyparsing, setuptools, six, tabulate, torchinfo, typing-inspection, urllib3, wfdb, and the torch-related packages axial_positional_embedding, CoLT5-attention, docstring-inheritance, hyper-connections, linear-attention-transformer, linformer, local-attention, product_key_memory, rotary-embedding-torch, torch-einops-utils.
- BSD (2- or 3-clause): contourpy, decorator, fsspec, h5py, idna, Jinja2, joblib, lazy-loader, MarkupSafe, mne, mne-bids, mpmath, networkx, pandas, pooch, pycparser, scikit-learn, skorch, soundfile, sympy, threadpoolctl, torchaudio.
- Apache-2.0: aiosignal, arrow, frozenlist, multidict, packaging (Apache-2.0 OR BSD-2-Clause), propcache, requests, tzdata, yarl, aiohttp (Apache-2.0 AND MIT), python-dateutil (dual Apache/BSD).
- PSF-2.0: aiohappyeyeballs, typing_extensions. matplotlib: PSF-based matplotlib license. Pillow: MIT-CMU.
- MPL-2.0: certifi, tqdm (MPL-2.0 AND MIT). Compatible with GPLv3 through MPL 2.0 section 3.3.
- Not determinable from metadata: **cycler** (metadata is only a copyright line) and **kiwisolver** (metadata is a separator line). Both are installed only through matplotlib, which the BCG code does not import. I believe they are BSD-licensed, but did not verify. Confirm before redistributing them.
- Development only: pytest 9.1.1 (MIT) in `requirements-dev.txt`.

No dependency was found to be incompatible with GPLv3.

## Bundled files

| File | What it is | License / status |
|---|---|---|
| `game/logo.svg` | "BCG" logo used in the game header | **Origin not determinable from the repository.** `../design/` holds a related `BCG-Logo.ai`. Confirm it is the authors' own work, or name its license. |
| `game/css/style.css` | Game styling | No third-party notice found. It names the fonts "Inter", "Segoe UI", "Helvetica Neue", Arial as fallbacks but does not load or bundle any font (no `@font-face`, no font link). |
| `game/js/audio.js` | Note frequencies played with the Web Audio API | Sound is synthesised in the browser; no audio file is bundled. The melody looks like Beethoven's "Ode to Joy" (public domain composition); I did not verify this. |
| PyQt `QFont("Segoe UI")`, `QFont("Courier New")` | Fonts requested from the operating system | Not bundled. |
| `models/eegnet_finetuned_mimed.pth` (and `../model/weights/*.pth`) | EEGNet weights trained in `../model/*.ipynb` | No longer tracked in git (they remain in earlier commits). The notebooks name the training data: **BCI Competition IV Dataset 2a** and the **MIMED** dataset from Mendeley Data. **Their licenses are not stated in the repository.** Check that the dataset terms allow redistributing weights derived from them before publishing the weights. |

## External services

- **Emotiv Cortex API** (`wss://localhost:6868`): the user's own Emotiv account, client id and secret are needed for real recordings. No Emotiv code is included. Emotiv's terms were not reviewed here.
