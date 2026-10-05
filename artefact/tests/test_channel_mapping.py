"""Cortex packet -> 14 electrode channels (docs layout: COUNTER, INTERPOLATED, AF3..AF4, RAW_CQ, ...)."""
import json

import numpy as np

from bcg_core.cortex_reader import (ELECTRODES, CortexReader, electrode_indices,
                                    extract_electrodes, eeg_columns, stream_names)
from conftest import make_app_config

COLS = ["COUNTER", "INTERPOLATED", *ELECTRODES, "RAW_CQ", "MARKER_HARDWARE", "MARKERS"]


def fake_packet(counter=7):
    """Electrode k carries 100+k; INTERPOLATED = 1 and RAW_CQ = 4 are recognisable sentinels."""
    return [counter, 1, *[100.0 + k for k in range(14)], 4, 0, []]


def subscribe_result(cols=COLS):
    return {"success": [{"cols": cols, "sid": "abc", "streamName": "eeg"}], "failure": []}


def test_electrode_order_is_documented_montage():
    assert ELECTRODES == ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                          "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def test_by_name_selection_includes_af4_and_skips_interpolated():
    idx, how = electrode_indices(COLS, 14)
    assert idx == list(range(2, 16)) and "by name" in how
    x = extract_electrodes(fake_packet(), idx, 14)
    assert x.dtype == np.float32 and x.shape == (14,)
    assert x.tolist() == [100.0 + k for k in range(14)]   # no INTERPOLATED (1.0), AF4 (113) present


def test_by_name_selection_follows_cols_order():
    cols = ["COUNTER", "INTERPOLATED", *reversed(ELECTRODES), "RAW_CQ"]
    idx, _ = electrode_indices(cols, 14)
    packet = [0, 0, *[200.0 + k for k in range(14)], 4]           # packet follows cols (reversed)
    x = extract_electrodes(packet, idx, 14)
    assert x[0] == 200.0 + 13 and x[13] == 200.0                  # AF3 is the last column here


def test_fallback_when_cols_missing_or_incomplete():
    for cols in (None, [], ["COUNTER", "INTERPOLATED", "AF3"]):
        idx, how = electrode_indices(cols, 14)
        assert idx is None and "fallback slice [2:16]" in how
    x = extract_electrodes(fake_packet(), None, 14)
    assert x.tolist() == [100.0 + k for k in range(14)]


def test_fewer_channels_take_first_electrodes():
    idx, _ = electrode_indices(COLS, 4)
    assert extract_electrodes(fake_packet(), idx, 4).tolist() == [100.0, 101.0, 102.0, 103.0]


def test_subscribe_result_parsing():
    assert stream_names(subscribe_result()) == ["eeg"]
    assert stream_names({"success": ["eeg"]}) == ["eeg"]          # plain-string form still accepted
    assert eeg_columns(subscribe_result()) == COLS
    assert eeg_columns({"success": ["eeg"]}) is None


def _reader_with_sink():
    reader = CortexReader(make_app_config())
    got = []
    reader.on_sample = got.append
    return reader, got


def test_reader_end_to_end_with_fake_messages():
    reader, got = _reader_with_sink()
    reader._pending_requests[9] = "subscribe"
    reader._on_message(None, json.dumps({"id": 9, "result": subscribe_result()}))
    assert reader.connected and reader._eeg_indices == list(range(2, 16))
    reader._on_message(None, json.dumps({"eeg": fake_packet(), "sid": "abc", "time": 1.0}))
    assert got[0].tolist() == [100.0 + k for k in range(14)]


def test_reader_uses_fallback_when_subscribe_has_no_cols():
    reader, got = _reader_with_sink()
    reader._pending_requests[9] = "subscribe"
    reader._on_message(None, json.dumps({"id": 9, "result": {"success": ["eeg"]}}))
    assert reader.connected and reader._eeg_indices is None
    reader._on_message(None, json.dumps({"eeg": fake_packet()}))
    assert got[0].tolist() == [100.0 + k for k in range(14)]
