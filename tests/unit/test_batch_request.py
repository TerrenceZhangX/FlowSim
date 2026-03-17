import asyncio
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock
import importlib.util
import pytest
import os

sys.path.insert(0, os.path.abspath("/flowsim/workload/framework/sglang/python"))

import sglang.bench_serving as bs


class DummyTokenizer:
    def get_vocab(self):
        return {str(i): i for i in range(100)}

    def encode(self, text, add_special_tokens=False):
        return [
            int(c) for c in text if c.isdigit() and int(c) in self.get_vocab()
        ]

    def decode(self, tokens):
        return "".join(str(t) for t in tokens if str(t) in self.get_vocab())


def make_requests(n):
    return [
        SimpleNamespace(
            prompt=f"prompt-{i}",
            prompt_len=10,
            output_len=5,
            image_data=None,
            text_prompt_len=10,
            vision_prompt_len=0,
            timestamp=0.0,
        )
        for i in range(n)
    ]


def _inject_min_args(**overrides):
    # minimal args to satisfy references inside benchmark()
    base = {
        "warmup_requests": 0,
        "backend": "mock",
        "dataset_name": "defined-len",
        "sharegpt_output_len": None,
        "random_input_len": None,
        "random_output_len": None,
        "random_range_ratio": None,
        "output_file": None,
        "output_details": False,
        # keep tokenization-related defaults (not used by fake_request_func)
        "disable_stream": False,
        "disable_ignore_eos": False,
        "return_logprob": False,
    }
    base.update(overrides)
    bs.set_global_args(SimpleNamespace(**base))


@pytest.fixture(autouse=True)
def _mock_requests_get(monkeypatch):
    """Prevent benchmark() from making real HTTP calls to get_server_info."""
    fake_resp = MagicMock()
    fake_resp.status_code = 404
    fake_resp.json.return_value = {}
    monkeypatch.setattr("requests.get", lambda *a, **kw: fake_resp)


def test_batched_requests_single_call():

    # prepare
    input_requests = make_requests(4)
    called = []

    async def fake_request_func(request_func_input, pbar=None):
        # Record call
        called.append(request_func_input)
        # Return a simple object containing benchmark expected fields
        return SimpleNamespace(
            success=True,
            prompt_len=getattr(request_func_input, "prompt_len", 0),
            output_len=getattr(request_func_input, "output_len", 0),
            generated_text="",
            error="",
            ttft=0.0,
            itl=[],  # must be a list
            latency=0.0,  # required by calculate_metrics
        )

    # Monkeypatch ASYNC_REQUEST_FUNCS for backend 'sglang'
    bs.ASYNC_REQUEST_FUNCS["mock"] = fake_request_func

    # Inject minimal global args used inside benchmark()
    _inject_min_args()

    # Run benchmark with batched_requests=True and no warmup (warmup_requests=0)
    asyncio.run(
        bs.benchmark(
            backend="mock",
            api_url="http://fake/api",
            base_url="http://fake",
            model_id="mymodel",
            tokenizer=DummyTokenizer(),
            input_requests=input_requests,
            request_rate=1.0,
            max_concurrency=None,
            disable_tqdm=True,
            lora_names=[],
            lora_request_distribution=None,
            lora_zipf_alpha=1.0,
            extra_request_body={},
            profile=False,
            pd_separated=False,
            flush_cache=False,
            warmup_requests=0,
            batched_requests=True,
        )
    )

    # assertions
    assert (
        len(called) == 1
    ), "batched requests should invoke the request func once"
    req_input = called[0]
    # In your new logic, prompt should be a list
    assert isinstance(
        req_input.prompt, list
    ), "prompt must be a list for batched_requests"
    assert req_input.prompt == [r.prompt for r in input_requests]
    # Prompt list length is len(input_requests)
    assert len(req_input.prompt) == len(input_requests)
    # Prompt_len and output_len is the length of the first request
    assert req_input.prompt_len == input_requests[0].prompt_len
    assert req_input.output_len == input_requests[0].output_len


def test_non_batched_requests_multiple_calls():

    input_requests = make_requests(3)
    called = []

    async def fake_request_func(request_func_input, pbar=None):
        called.append(request_func_input)
        return SimpleNamespace(
            success=True,
            prompt_len=getattr(request_func_input, "prompt_len", 0),
            output_len=getattr(request_func_input, "output_len", 0),
            generated_text="",
            error="",
            ttft=0.0,
            itl=[],  # must be a list
            latency=0.0,  # required by calculate_metrics
        )

    bs.ASYNC_REQUEST_FUNCS["mock"] = fake_request_func

    # Inject minimal global args used inside benchmark()
    _inject_min_args()

    asyncio.run(
        bs.benchmark(
            backend="mock",
            api_url="http://fake/api",
            base_url="http://fake",
            model_id="mymodel",
            tokenizer=DummyTokenizer(),
            input_requests=input_requests,
            request_rate=1.0,
            max_concurrency=None,
            disable_tqdm=True,
            lora_names=[],
            lora_request_distribution=None,
            lora_zipf_alpha=1.0,
            extra_request_body={},
            profile=False,
            pd_separated=False,
            flush_cache=False,
            warmup_requests=0,
            batched_requests=False,
        )
    )

    assert len(called) == len(
        input_requests
    ), "non-batched should call once per input request"
    # The first call's prompt should be a single string, not a list
    assert not isinstance(called[0].prompt, list)
    # Each call's prompt should match the corresponding input request
    for call, req in zip(called, input_requests):
        assert call.prompt == req.prompt
        assert call.prompt_len == req.prompt_len
        assert call.output_len == req.output_len
