# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional live-server parity check for an already packaged and served TDT decoder."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from nemo.collections.asr.parts.submodules.tdt_decoder import TDTDecoder


@pytest.mark.run_only_on("GPU")
def test_live_tdt_triton():
    url = os.environ.get("NEMO_TDT_TRITON_URL")
    artifact = os.environ.get("NEMO_TDT_DECODER_ARTIFACT")
    if not url or not artifact or not torch.cuda.is_available():
        pytest.skip("Set NEMO_TDT_TRITON_URL and NEMO_TDT_DECODER_ARTIFACT with CUDA available.")
    name = os.environ.get("NEMO_TDT_TRITON_MODEL", "tdt_decoder")
    endpoint = f"{url.rstrip('/')}/v2/models/{name}"
    decoder = TDTDecoder.load(artifact, "cuda", True)
    rng = np.random.default_rng(42)

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(endpoint + path, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as response:
            return json.load(response)

    def infer(encoded, lengths):
        payload = {
            "inputs": [
                {
                    "name": "ENCODER_OUTPUT",
                    "datatype": "FP32",
                    "shape": list(encoded.shape),
                    "data": encoded.ravel().tolist(),
                },
                {"name": "ENCODED_LENGTH", "datatype": "INT64", "shape": [len(lengths), 1], "data": lengths.tolist()},
            ]
        }
        result = request("/infer", payload)
        return {o["name"]: np.asarray(o["data"], dtype=np.int64).reshape(o["shape"]) for o in result["outputs"]}

    def case(batch, frames, lengths):
        encoded = rng.standard_normal((batch, decoder.joint.enc.in_features, frames)).astype(np.float32)
        lengths = np.asarray(lengths, dtype=np.int64)
        expected = decoder(torch.from_numpy(encoded), torch.from_numpy(lengths))
        return encoded, lengths, tuple(t.cpu().numpy().copy() for t in expected)

    def check(actual, expected):
        np.testing.assert_array_equal(actual["TOKEN_LENGTH"][:, 0], expected[1])
        for name, values, padding in [
            ("TOKENS", expected[0], -1),
            ("TIMESTAMPS", expected[2], -1),
            ("TOKEN_DURATIONS", expected[3], 0),
        ]:
            for batch, count in enumerate(expected[1]):
                np.testing.assert_array_equal(actual[name][batch, :count], values[batch, :count])
                assert np.all(actual[name][batch, count:] == padding)

    for args in [(1, 4, [4]), (2, 401, [401, 0]), (3, 19, [19, 9, 0]), (1, 4, [0])]:
        encoded, lengths, expected = case(*args)
        check(infer(encoded, lengths), expected)

    cases = [case(1, 23, [i + 1]) for i in range(16)]
    before = request("/stats")["model_stats"][0]
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda item: infer(item[0], item[1]), cases))
    for result, (_, _, expected) in zip(results, cases):
        check(result, expected)
    after = request("/stats")["model_stats"][0]
    assert int(after["inference_count"]) - int(before["inference_count"]) == 16
    if os.environ.get("NEMO_TDT_TRITON_REQUIRE_BATCHING") == "1":
        assert int(after["execution_count"]) - int(before["execution_count"]) < 16

    encoded, lengths, expected = cases[0]
    with pytest.raises(HTTPError) as error:
        infer(encoded, np.asarray([-1], dtype=np.int64))
    assert error.value.code == 400
    assert "Encoded lengths" in error.value.read().decode()
    check(infer(encoded, lengths), expected)
