let tiles = [];
let isRunning = false;
let frameId = null;


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
let nextSpawnAt = SPAWN_FRAMES;

function spawnNext() {
  if (trialIdx >= trialPool.length) {
    stopGame();
    return;
  }
  const lane = trialPool[trialIdx++];
  tiles.push(createTile(lane));
  nextSpawnAt = frameCount + TRIAL_GAP_FRAMES;
}
// <---


function update() {
    frameCount++;
    if (frameCount === nextSpawnAt)
        spawnNext();

    tiles.forEach(t => {
        t.y += TILE_SPEED;

        if (!t.played && t.y >= HIT_Y) {
            t.played = true;
            playNote();
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
    frameCount = 0;
    nextSpawnAt = SPAWN_FRAMES;
    tiles = [];
    isRunning = true;
    initAudio();
    spawnNext();
    loop();
}

function stopGame() {
  isRunning = false;
  cancelAnimationFrame(frameId);
}


document.getElementById('btn-start').addEventListener('click', startGame);
