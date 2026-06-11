let ws = null;

function setInfo(id, val) {
  const el = document.getElementById(id);
  if (el) 
    el.textContent = val;
}

// --->
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log('Connected');
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-connect').textContent = 'Disconnect';
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleServerMsg(msg);
  };

  ws.onerror = () => console.error('WebSocket error');

  ws.onclose = () => {
    console.log('Disconnected');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-connect').textContent = 'Connect';
    ws = null;
  };
}
// <---


// --->
function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify(obj));
}
// <---

// --->
function handleServerMsg(msg) {
  if (msg.type === 'status' || msg.type === 'phase_change') {
    const badge = document.getElementById('phase-badge');
    const overlay = document.getElementById('overlay-calibrating');

    if (!badge) return;

    if (overlay) overlay.style.display = 'none';

    if (msg.phase === 'collection') {
      badge.textContent = 'COLLECTION';
      badge.style.background = '#FFD700';
    } else if (msg.phase === 'calibrating') {
      badge.textContent = 'CALIBRATING';
      badge.style.background = '#FFA500';
      if (typeof stopGame === 'function') stopGame();
      if (overlay) overlay.style.display = 'flex';
    } else if (msg.phase === 'inference') {
      badge.textContent = 'INFERENCE';
      badge.style.background = '#2E8B57';
      document.getElementById('row-predict').style.display = 'flex';
    } else {
      badge.textContent = 'IDLE';
      badge.style.background = '#555';
    }

    setInfo('info-phase', (msg.phase || 'idle').toUpperCase()); 
    
    console.log('Phase:', msg.phase);
    return;
  }

  if (msg.type === 'trial_count') {
    setInfo('info-count', msg.count);
    console.log('Trials collected:', msg.count);
    if (typeof onTrialCountUpdate === 'function') onTrialCountUpdate(msg.count);
    return;
  }

  if (msg.type === 'calibration_done') {
    console.log('Calibration done, starting inference');
    send({ type: 'start_inference' });
    return;
  }

  if (msg.type === 'session_saved') {
    console.log('Session saved:', msg.path, 'trials=', msg.trials);
    setInfo('info-phase', 'SAVED');
    setInfo('info-trial', '—');
    return;
  }

  if (msg.type === 'prediction') {
    const conf = (msg.confidence * 100).toFixed(0);
    setInfo('info-predict', `${msg.label}  ${conf}%`); 
    console.log('Prediction:', msg.label, msg.confidence);
    handlePrediction(msg.label);
    return;
  }

  if (msg.type === 'error') {
    console.error('Server error:', msg.message);
    return;
  }

  if (msg.type === 'eeg_window') {
    if (msg.data?.length) {
        const vals = msg.data[0].slice(0, 4).map(v => v.toFixed(1)).join('  ');
        setInfo('info-eeg', vals); // ← thêm
    }
    return;
  }

  console.log('Unknown message', msg);
}
// <---

document.getElementById('btn-connect').addEventListener('click', () => {
  if (ws) ws.close();
  else connect();
});