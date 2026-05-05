const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

let predictedLane = null;

function setPredictedLane(laneIdx) {
  predictedLane = laneIdx;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}


function drawTile(lane, y) {
    const color = LANE_COLORS[lane];
    const x = lane * LANE_W + 8;

    ctx.fillStyle = color + 'cc';
    roundRect(x, y, TILE_W, TILE_H, TILE_RADIUS);
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 8;
    roundRect(x, y, TILE_W, TILE_H, TILE_RADIUS);
    ctx.stroke();
    ctx.shadowBlur  = 0;
}


function drawFrame(tiles) {
  // Clear
  ctx.fillStyle = '#0d0d18';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Lane tint
  LANE_COLORS.forEach((color, i) => {
    const grad = ctx.createLinearGradient(0, CANVAS_H * 0.5, 0, CANVAS_H);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(1, color + '18');
    ctx.fillStyle = grad;
    ctx.fillRect(i * LANE_W, 0, LANE_W, CANVAS_H);
  });

  // Lane divider
  ctx.strokeStyle = '#1e1e2e';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(LANE_W, 0);
  ctx.lineTo(LANE_W, CANVAS_H);
  ctx.stroke();

  // Tiles
  tiles.forEach(t => drawTile(t.lane, t.y));

  // Hit line
  LANE_COLORS.forEach((color, i) => {
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 14;
    ctx.fillRect(i * LANE_W + 8, HIT_Y, TILE_W, HIT_H);
    ctx.shadowBlur  = 0;
  });

  // Highlight predicted lane at hit line
  if (predictedLane !== null) {
    const i = predictedLane;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.strokeRect(i * LANE_W + 6, HIT_Y - 4, TILE_W + 4, HIT_H + 8);
  }

}


