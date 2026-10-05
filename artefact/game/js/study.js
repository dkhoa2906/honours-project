// SPDX-License-Identifier: GPL-3.0-or-later
// Study settings: how many labeled trials a session collects before it is saved.
//
// Each tile gives one motor trial (Left or Right) followed by one Rest trial, so
//   study1    78 trials  = 39 motor + 39 Rest. The setting used in Study 1 (default).
//   full_pool 104 trials = 52 motor + 52 Rest: the whole pool of 26 tiles per lane.
// Pick a preset with the page address, e.g. index.html?preset=full_pool. This reads the
// address only; the game still accepts no keyboard, mouse or touch input.
const STUDY_PRESETS = {
    study1: 78,
    full_pool: 104,
};
const DEFAULT_PRESET = 'study1';

const STUDY_PRESET = (() => {
    const wanted = new URLSearchParams(window.location.search).get('preset');
    if (wanted === null) return DEFAULT_PRESET;
    if (wanted in STUDY_PRESETS) return wanted;
    console.warn(`Unknown preset "${wanted}", using "${DEFAULT_PRESET}"`);
    return DEFAULT_PRESET;
})();

// Number of labeled trials (motor + Rest) at which the session is saved.
const LABELED_TRIALS = STUDY_PRESETS[STUDY_PRESET];
