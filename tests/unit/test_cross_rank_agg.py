"""Unit tests for utils.cross_rank_agg module."""

import csv
import json
import os
import tempfile

import pytest

from utils.cross_rank_agg import (
    aggregate,
    classify_kernel,
    is_comm,
    print_result,
)

# ---------------------------------------------------------------------------
# classify_kernel
# ---------------------------------------------------------------------------

_CLASSIFY_CASES = [
    # (kernel_name, expected_category)
    ("ncclKernel_AllReduce_RING_LL_Sum_float", "all_reduce"),
    ("cross_device_reduce_1block", "all_reduce"),
    ("ncclDevAllGatherCollNet", "all_gather"),
    ("ncclKernel_ReduceScatter_RING", "reduce_scatter"),
    ("ncclKernel_AllToAll", "all_to_all"),
    ("fused_moe_kernel", "moe"),
    ("deep_gemm_fp8_kernel", "gemm_fp8"),
    ("flash_fwd_kernel", "attention"),
    ("sm90_xmma_gemm", "attention"),
    ("fused_add_rmsnorm_kernel", "rmsnorm"),
    ("per_token_quant_int8", "quantize"),
    ("topk_softmax_kernel", "topk_gating"),
    ("moe_sum_reduce", "moe_misc"),
    ("argmax_kernel", "sampler"),
    ("CatArrayBatchedCopy", "copy"),
    ("some_unknown_kernel", "other"),
]


@pytest.mark.parametrize("name,expected", _CLASSIFY_CASES)
def test_classify_kernel(name, expected):
    assert classify_kernel(name) == expected


def test_classify_kernel_second_pass_source():
    """Second-pass classification via source code field."""
    assert (
        classify_kernel("some_generic_kernel", source="fused_moe_impl")
        == "moe_misc"
    )


def test_classify_kernel_second_pass_callstack():
    """Second-pass classification via callstack field."""
    assert (
        classify_kernel("generic_kernel", callstack="sampler <- forward")
        == "sampler"
    )


# ---------------------------------------------------------------------------
# is_comm
# ---------------------------------------------------------------------------


def test_is_comm():
    assert is_comm("all_reduce") is True
    assert is_comm("all_gather") is True
    assert is_comm("reduce_scatter") is True
    assert is_comm("all_to_all") is True
    assert is_comm("moe") is False
    assert is_comm("other") is False


# ---------------------------------------------------------------------------
# Helper: write temporary CSV files
# ---------------------------------------------------------------------------
_HEADER = [
    "Name",
    "Dims",
    "Data Type",
    "Input/Output",
    "Descriptions",
    "Duration (us)",
    "op",
    "operation",
    "Source Code",
    "Call Stack",
]


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADER)
        writer.writeheader()
        for row in rows:
            full = {h: "" for h in _HEADER}
            full.update(row)
            writer.writerow(full)


def _make_kernel_row(name: str, duration: float) -> dict:
    return {"Name": name, "Duration (us)": str(duration)}


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    """Tests for cross_rank_agg.aggregate()."""

    def test_single_rank_compute_only(self, tmp_path):
        """Single rank, compute kernels only."""
        csv_path = str(tmp_path / "TP-0-DECODE.csv")
        _write_csv(
            csv_path,
            [
                _make_kernel_row("fused_moe_kernel", 100.0),
                _make_kernel_row("flash_fwd_kernel", 200.0),
                _make_kernel_row("fused_moe_kernel", 50.0),
            ],
        )

        result = aggregate(csv_files=[csv_path], stage="DECODE")
        assert result["num_ranks"] == 1
        assert result["stage"] == "DECODE"
        assert result["categories"]["moe"]["us"] == 150.0
        assert result["categories"]["attention"]["us"] == 200.0
        assert result["total_kernel_us"] == 350.0

    def test_multi_rank_symmetric_comm_min(self, tmp_path):
        """Symmetric comm (all_reduce) uses per-invocation min."""
        # Rank 0: two all_reduce calls with durations [100, 200]
        csv0 = str(tmp_path / "TP-0-DECODE.csv")
        _write_csv(
            csv0,
            [
                _make_kernel_row("cross_device_reduce_1block", 100.0),
                _make_kernel_row("cross_device_reduce_1block", 200.0),
            ],
        )
        # Rank 1: two all_reduce calls with durations [150, 80]
        csv1 = str(tmp_path / "TP-1-DECODE.csv")
        _write_csv(
            csv1,
            [
                _make_kernel_row("cross_device_reduce_1block", 150.0),
                _make_kernel_row("cross_device_reduce_1block", 80.0),
            ],
        )

        result = aggregate(csv_files=[csv0, csv1], stage="DECODE")
        # Per-invocation min: min(100,150) + min(200,80) = 100 + 80 = 180
        assert result["categories"]["all_reduce"]["us"] == 180.0
        assert (
            result["categories"]["all_reduce"]["method"] == "per-invocation-min"
        )

    def test_multi_rank_asymmetric_comm_max(self, tmp_path):
        """Asymmetric comm (all_to_all) uses max-across-ranks."""
        csv0 = str(tmp_path / "TP-0-DECODE.csv")
        _write_csv(
            csv0,
            [
                _make_kernel_row("ncclKernel_AllToAll", 500.0),
            ],
        )
        csv1 = str(tmp_path / "TP-1-DECODE.csv")
        _write_csv(
            csv1,
            [
                _make_kernel_row("ncclKernel_AllToAll", 800.0),
            ],
        )

        result = aggregate(csv_files=[csv0, csv1], stage="DECODE")
        assert result["categories"]["all_to_all"]["us"] == 800.0
        assert (
            result["categories"]["all_to_all"]["method"] == "max-across-ranks"
        )

    def test_compute_mean_across_ranks(self, tmp_path):
        """Compute kernels use mean across ranks."""
        csv0 = str(tmp_path / "TP-0-DECODE.csv")
        _write_csv(csv0, [_make_kernel_row("fused_moe_kernel", 100.0)])
        csv1 = str(tmp_path / "TP-1-DECODE.csv")
        _write_csv(csv1, [_make_kernel_row("fused_moe_kernel", 200.0)])

        result = aggregate(csv_files=[csv0, csv1], stage="DECODE")
        assert result["categories"]["moe"]["us"] == 150.0
        assert result["categories"]["moe"]["method"] == "mean-across-ranks"

    def test_compute_only_flag(self, tmp_path):
        """compute_only=True excludes communication kernels."""
        csv_path = str(tmp_path / "TP-0-DECODE.csv")
        _write_csv(
            csv_path,
            [
                _make_kernel_row("fused_moe_kernel", 100.0),
                _make_kernel_row("cross_device_reduce_1block", 500.0),
            ],
        )

        result = aggregate(
            csv_files=[csv_path], stage="DECODE", compute_only=True
        )
        assert "all_reduce" not in result["categories"]
        assert result["categories"]["moe"]["us"] == 100.0

    def test_csv_dir_discovery(self, tmp_path):
        """aggregate(csv_dir=...) discovers CSVs by stage pattern."""
        csv_path = str(tmp_path / "rank0-DECODE.csv")
        _write_csv(csv_path, [_make_kernel_row("flash_fwd_kernel", 50.0)])
        # Also create an EXTEND CSV that should NOT be picked up
        extend_path = str(tmp_path / "rank0-EXTEND.csv")
        _write_csv(extend_path, [_make_kernel_row("flash_fwd_kernel", 999.0)])

        result = aggregate(csv_dir=str(tmp_path), stage="DECODE")
        assert result["categories"]["attention"]["us"] == 50.0

    def test_empty_csv_dir(self, tmp_path):
        """Empty directory returns empty dict."""
        result = aggregate(csv_dir=str(tmp_path), stage="DECODE")
        assert result == {}

    def test_no_args_raises(self):
        """Must provide either csv_dir or csv_files."""
        with pytest.raises(ValueError):
            aggregate()

    def test_print_result_no_crash(self, tmp_path, capsys):
        """print_result should not crash on valid input."""
        csv_path = str(tmp_path / "TP-0-DECODE.csv")
        _write_csv(csv_path, [_make_kernel_row("flash_fwd_kernel", 50.0)])
        result = aggregate(csv_files=[csv_path], stage="DECODE")
        print_result(result)
        captured = capsys.readouterr()
        assert "attention" in captured.out

    def test_print_result_empty(self, capsys):
        """print_result on empty dict should not crash."""
        print_result({})
        captured = capsys.readouterr()
        assert captured.out == ""
