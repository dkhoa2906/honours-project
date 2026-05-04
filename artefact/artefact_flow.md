# Project Context: EEG-Based Motor Imagery Calibration

## 1. High-Level Goal

The project's primary objective is to develop and evaluate a more effective and engaging method for calibrating EEG-based motor imagery (MI) models. It compares two approaches:

1.  **Traditional Method:** A standard, cue-based data collection paradigm.
2.  **Game Method:** A novel, interactive rhythm game designed to make the calibration process less tedious and potentially more effective.

The final output is a comparison of models calibrated using data from these two methods to determine which is better.

## 2. Core Components & Artefacts

The `artefact/` directory contains three key components:

*   **`model/`**:
    *   Contains a pre-trained 2-class (left/right hand) EEGNet model built with PyTorch (`.pth` files).
    *   Includes Jupyter Notebooks (`.ipynb`) that detail the entire training and fine-tuning pipeline, from pre-training on the BCI IV 2a dataset to fine-tuning on the MIMED dataset.
    *   **Key File:** `02_finetune_mimed.ipynb` is crucial as it defines the exact "calibration" process: fine-tuning the model on new data for 150 epochs.

*   **`data_test_and_collect/`**:
    *   A set of Python scripts for handling EEG data.
    *   `cortex_reader.py`: Interfaces with the EEG hardware.
    *   `classifier.py`: Contains logic for real-time prediction using a trained model, including preprocessing steps (filtering, normalization).
    *   This represents the "traditional" data collection method and provides the building blocks for the server-side logic.

*   **`game/`**:
    *   Contains the web-based rhythm game, `bcg-rhythm.html`.
    *   This is a self-contained HTML file with all necessary CSS and JavaScript.
    *   Currently, it simulates MI input using keyboard arrow keys and logs the trial data to the browser console.
    *   The game has a fixed "note chart" of 40 trials that takes approximately 1 minute to complete at 70 BPM.

## 3. The User's Objective & Proposed Solution

The user wants to evolve the game from a simple data logger into a truly interactive BCI system. This led to the idea of an **"online adaptive calibration"** model.

We have agreed on a solution called **Option A (Server-Side Calibration)**. This approach avoids the impracticality of running the heavy PyTorch calibration in the browser.

### The "Option A" Workflow:

1.  **Connect:** A new Python script, `server.py`, will run a WebSocket server. The `bcg-rhythm.html` game, when opened in a browser, will connect to this server.

2.  **Stream EEG:** The Python server will use the existing `cortex_reader.py` logic to get a live EEG data stream and send it, sample by sample, to the game via the WebSocket.

3.  **Phase 1: Data Collection (The "Dumb" Part):**
    *   The game starts in "collection mode."
    *   It shows the falling notes as cues. As the user performs motor imagery, the game records the corresponding chunk of EEG data it's receiving from the server.
    *   This phase is expected to last **~1 minute** to collect the 40 trials from the note chart.

4.  **Phase 2: Calibration (The "Waiting" Part):**
    *   The game sends the entire block of `(cue, eeg_data)` pairs back to the Python server.
    *   The game UI displays a "Calibrating..." message.
    *   The Python server receives the data and uses the logic from `02_finetune_mimed.ipynb` to fine-tune the pre-trained EEGNet model.
    *   This is expected to take **~1-5 minutes**, depending heavily on whether the server is running on a GPU or CPU.

5.  **Phase 3: Interactive Gameplay:**
    *   Once calibration is complete, the Python server switches to "inference mode."
    *   It now processes the live EEG stream with the newly calibrated model and sends *predictions* (e.g., "LEFT", "RIGHT") to the game.
    *   The game logic is updated to use these predictions as its input, creating a real-time, interactive BCI experience.

## 4. Current Status & Next Step

We have fully defined the plan (Option A) and are ready to begin implementation.

**The immediate next step is to set up the Python WebSocket server.** This involves:
1.  Creating a `server.py` file in the `artefact/data_test_and_collect/` directory.
2.  Installing the `websockets` Python library into the project's virtual environment (`venv/`). The user has encountered issues with the agent running `pip` commands, so this may require manual user intervention.
3.  Writing the basic server code to accept WebSocket connections.