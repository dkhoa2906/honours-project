
import logging
import websocket
import threading
import json
import ssl

import numpy as np

from typing import Optional
from bcg_core.config_schema import AppConfig


logger = logging.getLogger(__name__)


class CortexReader:
    def __init__(self, config: AppConfig):
        self.connected = False
        self._config = config

        self._ws = None
        self.on_sample = None

        self._token = ""
        self._headset_id = ""
        self._session_id = ""

        logger.info("CortexReader created")


    def start(self):
        self._ws = websocket.WebSocketApp(
            "wss://localhost:6868",
            on_open = self._on_open,
            on_message = self._on_message,
            on_error = self._on_error,
            on_close = self._on_close
        )

        thread = threading.Thread(
            target=self._ws.run_forever, 
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True
        )
        thread.start()

        logger.info("CortexReader started")


    def stop(self):
        if self._ws:
            self._ws.close()
            self.connected = False
            logger.info("CortexReader stop")


    # Callbacks --->
    
    def _on_open(self, ws):
        logger.info("✓ WebSocket connected. Authorizing...")

        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "authorize",
            "params": {
                "clientId": self._config.cortex_api.client_id,
                "clientSecret": self._config.cortex_api.client_secret
            }
        }

        ws.send(json.dumps(msg))


    def _on_message(self, ws, raw):
        data = json.loads(raw)

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
        
        if req_id == 1:
            self._token = data["result"]["cortexToken"]
            logger.info("✓ Authorized. Querying headsets...")
            ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "queryHeadsets",
                "params": {} 
            }))
        elif req_id == 2:
            headsets = data.get("result", [])
            if not headsets:
                logger.error("No headset found")
                return
            self._headset_id = headsets[0]["id"]
            logger.info(f"✓ Headset: {self._headset_id} connected. Creating session...")
            ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "createSession",
                "params": {
                    "cortexToken": self._token,
                    "headset": self._headset_id,
                    "status": "active"
                }
            }))
        elif req_id == 3:
            self._session_id = data["result"]["id"]
            logger.info(f"✓ Session: {self._session_id}. Subscribing EEG...")
            ws.send(json.dumps({
                "jsonrpc": "2.0", 
                "id": 4,
                "method": "subscribe",
                "params": {
                    "cortexToken": self._token,
                    "session": self._session_id,
                    "streams": ["eeg"]
                }
            }))
        elif req_id == 4:
            self.connected = True
            logger.info("✓ EEG stream activated")


    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_code, close_msg):
        self.connected = False
        logger.warning(f"WebSocket closed: {close_code}")

    
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