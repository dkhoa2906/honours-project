let ws = null;

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
    // Implement later
}
// <---

document.getElementById('btn-connect').addEventListener('click', () => {
  if (ws) ws.close();
  else connect();
});