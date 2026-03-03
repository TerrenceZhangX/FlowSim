import pytest
import sys
import os

sys.path.insert(0, os.path.abspath("/flowsim/workload/framework/sglang/python"))
from sglang.bench_serving import generate_defined_len_requests


class DummyTokenizer:
    def get_vocab(self):
        return {str(i): i for i in range(100)}


def test_defined_len_requests_single_pair():
    tokenizer = DummyTokenizer()
    lens_str = "10:5"
    num_prompts = 2
    requests = generate_defined_len_requests(lens_str, num_prompts, tokenizer)
    assert all(r.prompt_len == 10 and r.output_len == 5 for r in requests)


def test_defined_len_requests_zero_prompts():
    tokenizer = DummyTokenizer()
    lens_str = "8:2"
    num_prompts = 0
    requests = generate_defined_len_requests(lens_str, num_prompts, tokenizer)
    assert requests == []


def test_defined_len_requests_invalid_lens_str():
    tokenizer = DummyTokenizer()
    with pytest.raises(Exception):
        generate_defined_len_requests("badformat", 1, tokenizer)
