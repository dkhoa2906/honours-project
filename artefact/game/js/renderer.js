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
    const grad = ctx.createLinearGradient(x, y, x, y + TILE_H);
    grad.addColorStop(0, color + "ee");
    grad.addColorStop(0.45, color + "c8");
    grad.addColorStop(1, color + "72");

    ctx.fillStyle = grad;
    roundRect(x, y, TILE_W, TILE_H, TILE_RADIUS);
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.8;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 16;
    roundRect(x, y, TILE_W, TILE_H, TILE_RADIUS);
    ctx.stroke();

    // Soft top sheen for a cleaner "card-like" look.
    ctx.fillStyle = "rgba(255,255,255,0.22)";
    roundRect(x + 6, y + 6, TILE_W - 12, 8, 4);
    ctx.fill();
    ctx.shadowBlur  = 0;
}


function drawFrame(tiles) {
  // Base background gradient
  const bg = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
  bg.addColorStop(0, "#1a2444");
  bg.addColorStop(0.5, "#111a33");
  bg.addColorStop(1, "#0b1023");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Subtle lane backplates
  ctx.fillStyle = "rgba(255,255,255,0.04)";
  ctx.fillRect(0, 0, LANE_W, CANVAS_H);
  ctx.fillRect(LANE_W, 0, LANE_W, CANVAS_H);

  // Lane tint
  LANE_COLORS.forEach((color, i) => {
    const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(0.65, color + '12');
    grad.addColorStop(1, color + '26');
    ctx.fillStyle = grad;
    ctx.fillRect(i * LANE_W, 0, LANE_W, CANVAS_H);
  });

  // Soft grid for presentation screenshots.
  ctx.strokeStyle = "rgba(190,210,255,0.08)";
  ctx.lineWidth = 1;
  for (let y = 28; y < CANVAS_H; y += 28) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(CANVAS_W, y);
    ctx.stroke();
  }

  // Lane divider
  ctx.strokeStyle = 'rgba(190,210,255,0.25)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(LANE_W, 0);
  ctx.lineTo(LANE_W, CANVAS_H);
  ctx.stroke();

  // Tiles
  tiles.forEach(t => drawTile(t.lane, t.y));

  // Hit line
  LANE_COLORS.forEach((color, i) => {
    const glow = ctx.createLinearGradient(0, HIT_Y - 6, 0, HIT_Y + HIT_H + 6);
    glow.addColorStop(0, "rgba(255,255,255,0.18)");
    glow.addColorStop(0.5, color);
    glow.addColorStop(1, color + "88");
    ctx.fillStyle   = glow;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 22;
    ctx.fillRect(i * LANE_W + 8, HIT_Y, TILE_W, HIT_H);
    ctx.shadowBlur  = 0;
  });

  // Highlight predicted lane at hit line
  if (predictedLane !== null) {
    const i = predictedLane;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 10;
    ctx.strokeRect(i * LANE_W + 6, HIT_Y - 4, TILE_W + 4, HIT_H + 8);
    ctx.shadowBlur = 0;
  }

}


