
import logging
import websocket
import threading
import json
import ssl
import time

import numpy as np

from typing import Optional
from bcg_core.config_schema import AppConfig


logger = logging.getLogger(__name__)


class CortexReader:
    def __init__(self, config: AppConfig):
        self.connected = False
        self._config = config

        self._ws = None
        self._ws_thread = None
        self.on_sample = None

        self._token = ""
        self._headset_id = ""
        self._session_id = ""
        self._pending_requests: dict[int, str] = {}
        self._next_rpc_id = 1
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._watchdog_timeout_sec = 8.0
        self.last_error: Optional[str] = None
        self.connection_state = "idle"

        logger.info("CortexReader created")


    def start(self):
        # Reset state on each start to avoid stale status from previous sessions.
        self.connected = False
        self._token = ""
        self._headset_id = ""
        self._session_id = ""
        self._pending_requests.clear()
        self._next_rpc_id = 1
        self.last_error = None
        self.connection_state = "starting"
        self._ready_event.clear()
        self._stop_event.clear()

        self._ws = websocket.WebSocketApp(
            "wss://localhost:6868",
            on_open = self._on_open,
            on_message = self._on_message,
            on_error = self._on_error,
            on_close = self._on_close
        )

        self._ws_thread = threading.Thread(
            target=self._ws.run_forever, 
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True
        )
        self._ws_thread.start()

        logger.info("CortexReader started")
        threading.Thread(target=self._connection_watchdog, daemon=True).start()


    def stop(self):
        self._stop_event.set()
        if self._ws:
            self._ws.close()
        self.connected = False
        self.connection_state = "stopped"
        logger.info("CortexReader stop")

    def wait_until_connected(self, timeout_sec: float = 3.0) -> bool:
        if self.connected:
            return True
        self._ready_event.wait(timeout=timeout_sec)
        return self.connected

    def get_status_message(self) -> str:
        if self.last_error:
            return self.last_error
        return f"Cortex state: {self.connection_state}"


    # Callbacks --->
    
    def _on_open(self, ws):
        logger.info("✓ WebSocket connected. Authorizing...")
        self._set_state("socket_connected")
        self._request_access()


    def _on_message(self, ws, raw):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._mark_error(f"Received non-JSON message from Cortex: {raw!r}")
            return

        if "eeg" not in data:
            logger.info(f"Cortex message: {data}")
        if "error" in data:
            err = data["error"]
            req_method = self._pending_requests.pop(data.get("id"), None)
            self._handle_rpc_error(err, req_method=req_method)
            return

        # EEG data stream
        if "eeg" in data:
            sample = np.array(
                data["eeg"][1:self._config.model.n_channels+1], 
                dtype=np.float32
            )
        
            if self.on_sample:
                self.on_sample(sample)
            return
        
        # RPC responses
        req_id = data.get("id")
        if req_id is None and "warning" not in data:
            logger.debug(f"Cortex non-RPC message: {data}")
        if req_id is None:
            return

        req_method = self._pending_requests.pop(req_id, None)
        if req_method is None:
            logger.debug(f"Cortex RPC response id={req_id}: {data}")
            return

        result = data.get("result")
        if req_method == "requestAccess":
            # requestAccess can be denied or unsupported depending on Cortex version/policy.
            if isinstance(result, dict) and result.get("accessGranted") is False:
                self._mark_error(
                    "Cortex denied app access. Open Emotiv Launcher and grant this app permission."
                )
                return
            logger.info("✓ Access granted (or not required). Authorizing...")
            self._authorize()
        elif req_method == "authorize":
            if not result or "cortexToken" not in result:
                self._mark_error(
                    "Authorize response missing cortexToken. Verify client credentials and app permission."
                )
                return
            self._token = result["cortexToken"]
            self._set_state("authorized")
            logger.info("✓ Authorized. Querying headsets...")
            self._send_rpc("queryHeadsets", {})
        elif req_method == "queryHeadsets":
            headsets = result if isinstance(result, list) else []
            connected_headsets = [h for h in headsets if h.get("status") == "connected"]
            chosen = connected_headsets[0] if connected_headsets else (headsets[0] if headsets else None)
            if not chosen or "id" not in chosen:
                self._mark_error("No headset found. Connect and enable the headset in Emotiv Launcher.")
                return
            self._headset_id = chosen["id"]
            self._set_state("headset_selected")
            logger.info(f"✓ Headset: {self._headset_id}. Creating session...")
            self._send_rpc(
                "createSession",
                {
                    "cortexToken": self._token,
                    "headset": self._headset_id,
                    "status": "active",
                },
            )
        elif req_method == "createSession":
            if not result or "id" not in result:
                self._mark_error(
                    f"CreateSession failed or malformed response: {data}"
                )
                return
            self._session_id = result["id"]
            self._set_state("session_active")
            logger.info(f"✓ Session: {self._session_id}. Subscribing EEG...")
            self._send_rpc(
                "subscribe",
                {
                    "cortexToken": self._token,
                    "session": self._session_id,
                    "streams": ["eeg"],
                },
            )
        elif req_method == "subscribe":
            success_streams = []
            if isinstance(result, dict):
                success_streams = result.get("success", []) or result.get("streams", [])
            if not success_streams or "eeg" not in success_streams:
                self._mark_error(f"Subscribe did not confirm EEG stream: {data}")
                return
            self.connected = True
            self.last_error = None
            self._set_state("streaming")
            self._ready_event.set()
            logger.info("✓ EEG stream activated")


    def _on_error(self, ws, error):
        self._mark_error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_code, close_msg):
        self.connected = False
        if self._stop_event.is_set():
            logger.info("WebSocket closed by client")
            return
        self._mark_error(f"WebSocket closed: code={close_code} message={close_msg}")

    def _connection_watchdog(self):
        # Cortex should usually finish authorize->session->subscribe quickly.
        if self._ready_event.wait(timeout=self._watchdog_timeout_sec):
            return
        if self._stop_event.is_set() or self.connected:
            return
        if self.last_error:
            logger.error("Cortex connection failed: %s", self.last_error)
            return
        self._mark_error(
            "Cortex connection timeout while in state "
            f"'{self.connection_state}'. Check Emotiv login, headset status, app permission, "
            "and whether Cortex service is running."
        )

    def _request_access(self):
        self._set_state("requesting_access")
        self._send_rpc(
            "requestAccess",
            {
                "clientId": self._config.cortex_api.client_id,
                "clientSecret": self._config.cortex_api.client_secret,
            },
        )

    def _authorize(self):
        self._set_state("authorizing")
        self._send_rpc(
            "authorize",
            {
                "clientId": self._config.cortex_api.client_id,
                "clientSecret": self._config.cortex_api.client_secret,
            },
        )

    def _send_rpc(self, method: str, params: dict):
        if self._ws is None:
            self._mark_error(f"Cannot send {method}: websocket not initialized")
            return
        rpc_id = self._next_rpc_id
        self._next_rpc_id += 1
        self._pending_requests[rpc_id] = method
        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
            "params": params,
        }
        with self._send_lock:
            try:
                self._ws.send(json.dumps(payload))
            except Exception as exc:
                self._pending_requests.pop(rpc_id, None)
                self._mark_error(f"Failed to send Cortex RPC '{method}': {exc}")

    def _handle_rpc_error(self, err: dict, req_method: Optional[str]):
        code = err.get("code")
        message = err.get("message")
        data = err.get("data")
        base = f"Cortex RPC error on {req_method or 'unknown'}: code={code} message={message}"
        if data:
            base += f" data={data}"

        detail = str(message or "").lower()
        if req_method == "requestAccess" and "method not found" in detail:
            logger.warning("requestAccess unsupported by Cortex version; continuing with authorize")
            self._authorize()
            return
        if req_method in {"requestAccess", "authorize"} and any(
            token in detail for token in ["not authorized", "access", "permission", "deny"]
        ):
            self._mark_error(
                "Cortex authorization blocked. In Emotiv Launcher, log in and grant app permission, "
                "then retry."
            )
            return
        self._mark_error(base)

    def _set_state(self, state: str):
        with self._state_lock:
            self.connection_state = state

    def _mark_error(self, message: str):
        self.connected = False
        self.last_error = message
        self._set_state("error")
        logger.error(message)

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with open ("config/milive_conf.json") as f:
        config = AppConfig.model_validate(json.load(f))

    reader = CortexReader(config)

    def on_sample(sample):
        print(f"Sample: {sample.shape} {sample[:3]}")

    reader.on_sample = on_sample
    reader.start()

    import time
    time.sleep(5)
    reader.stop()