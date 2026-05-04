const MELODY = [
  // Phrase 1
  329.63, 329.63, 349.23, 392.00, // E E F G
  392.00, 349.23, 329.63, 293.66, // G F E D
  261.63, 261.63, 293.66, 329.63, // C C D E
  329.63, 293.66, 293.66,          // E D D

  // Phrase 2
  329.63, 329.63, 349.23, 392.00, // E E F G
  392.00, 349.23, 329.63, 293.66, // G F E D
  261.63, 261.63, 293.66, 329.63, // C C D E
  293.66, 261.63, 261.63,          // D C C
];

let audioCtx  = null;
let melodyIdx = 0;

// --->
function initAudio() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}

// --->
function resetMelody() { melodyIdx = 0; }


// --->
function playNote() {
  if (!audioCtx) return;

  const freq = MELODY[melodyIdx % MELODY.length];
  melodyIdx++;

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();

  osc.connect(gain);
  gain.connect(audioCtx.destination);

  osc.type = 'sine';
  osc.frequency.value = freq;

  const t = audioCtx.currentTime;
  gain.gain.setValueAtTime(0, t);
  gain.gain.linearRampToValueAtTime(0.5, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 1.2);

  osc.start(t);
  osc.stop(t + 1.2);
}
