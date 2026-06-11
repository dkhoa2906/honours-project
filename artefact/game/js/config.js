// Canvas
const CANVAS_W = 320;
const CANVAS_H = 520;

// Lanes
const N_LANES = 2;
const LANE_W = CANVAS_W / N_LANES;
const LANE_COLORS = ['#5591c7', '#a86fdf'];
const LANE_NAMES = ['LEFT', 'RIGHT'];

// Tiles
const TILE_W = LANE_W - 16;
const TILE_H = 1300;
const TILE_RADIUS = 6;
const TILE_SPEED = 3;

// Hit line
const HIT_Y = CANVAS_H - 80; 
const HIT_H = 4;

// Trial / collect
const TRIALS_PER_CLASS = 26;
const TRIAL_DURATION_MS = 4000;
const INTER_TRIAL_MS = 500;

// Rest
const REST_DEADZONE_MS = 500;
const REST_DURATION_MS = 4000;

// Timing
const SPAWN_FRAMES = Math.round((HIT_Y + TILE_H) / TILE_SPEED);
const TRIAL_GAP_FRAMES = Math.round(
    (TRIAL_DURATION_MS + REST_DEADZONE_MS + REST_DURATION_MS + 
        INTER_TRIAL_MS) / 1000 * 60);

// Server 
const WS_URL = 'ws://localhost:8765';