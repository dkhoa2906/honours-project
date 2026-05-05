let tiles = [];
let isRunning = false;
let frameId = null;

let totalTrials = 0;
const N_CLASSES = 3;
const MIN_TRIALS_TO_CALIBRATE = TRIALS_PER_CLASS * N_CLASSES; 
let calibrationRequested = false;


function createTile(lane) {
    return {
        lane,
        y: -TILE_H,
        played: false
    };
}


// Init trial pool --->
let trialPool = [];
let trialIdx = 0;

function buildTrialPool() {
    const pool = [];
    LANE_NAMES.forEach((name, lane) => {
        for (let i=0; i<TRIALS_PER_CLASS; i++)
            pool.push(lane);
    });

    for (let i=pool.length-1; i>0; i--) {
        const j = Math.floor(Math.random()*(i+1));
        [pool[i], pool[j]] = [pool[j], pool[i]]
    }
    return pool;
}
// <---


// Spawn --->
let frameCount = 0;  

function spawnNext() {
    if (trialIdx >= trialPool.length) { 
        stopGame(); 
        return; 
    }
    const lane = trialPool[trialIdx++];
    tiles.push(createTile(lane));
}
// <---

const labelMap = { LEFT: 'Left Hand', RIGHT: 'Right Hand' };

function update() {
    frameCount++;

    tiles.forEach(t => {
        t.y += TILE_SPEED;
        const tileCenterY = t.y + TILE_H / 2;
        
        if (!t.played && tileCenterY >= HIT_Y) {
            t.played = true;
            const label = labelMap[LANE_NAMES[t.lane]];
            send({ type: 'trial_start', label });
            setInfo('info-trial', label);
            playNote();

            setTimeout(() => {
                send({ type: 'trial_end', label });
                setTimeout(() => {
                    send({ type: 'trial_start', label: 'Rest' });
                    setInfo('info-trial', 'Rest');

                    setTimeout(() => {
                        send({ type: 'trial_end', label: 'Rest' });
                        setInfo('info-trial', '—');
                        setTimeout(() => spawnNext(), INTER_TRIAL_MS);
                    }, REST_DURATION_MS);
                }, REST_DEADZONE_MS);
            }, TRIAL_DURATION_MS);
        }
    });

    tiles = tiles.filter(t => t.y < HIT_Y + TILE_H)
}

function loop() {
    update();
    drawFrame(tiles);
    frameId = requestAnimationFrame(loop);
}

function startGame() {
    trialPool = buildTrialPool();
    trialIdx = 0;
    tiles = [];
    isRunning = true;
    initAudio();
    loop();
    setTimeout(() => spawnNext(), 2000);
}

function stopGame() {
  isRunning = false;
  cancelAnimationFrame(frameId);
}


document.getElementById('btn-start').addEventListener('click', startGame);

function onTrialCountUpdate(count) {
  totalTrials = count;
  if (!calibrationRequested && totalTrials >= MIN_TRIALS_TO_CALIBRATE) {
    calibrationRequested = true;
    send({ type: 'start_calibration' });
  }
}

function handlePrediction(label) {
  const labelToLane = {
    'Left Hand': 0,
    'Right Hand': 1
  };
  if (label in labelToLane) {
    setPredictedLane(labelToLane[label]);
  }
}