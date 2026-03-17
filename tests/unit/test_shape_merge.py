"""Unit tests for utils.shape_merge module."""

import csv
import os

import pytest

from utils.shape_merge import (
    merge_shapes,
    merge_shapes_dir,
    _rank_stage_key,
)

_CSV_HEADER = [
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            full = {h: "" for h in _CSV_HEADER}
            full.update(row)
            writer.writerow(full)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# _rank_stage_key
# ---------------------------------------------------------------------------


class TestRankStageKey:
    def test_simple_tp(self):
        assert _rank_stage_key("1772525862-TP-0-DECODE.trace.csv") == (
            "TP-0",
            "DECODE",
        )

    def test_tp_dp(self):
        assert _rank_stage_key("1772529412-TP-1-DP-1-EXTEND.trace.csv") == (
            "TP-1-DP-1",
            "EXTEND",
        )

    def test_tp_dp_ep(self):
        assert _rank_stage_key("123-TP-2-DP-0-EP-3-DECODE.trace.csv") == (
            "TP-2-DP-0-EP-3",
            "DECODE",
        )

    def test_no_match(self):
        assert _rank_stage_key("random_file.csv") is None


# ---------------------------------------------------------------------------
# merge_shapes (single pair)
# ---------------------------------------------------------------------------


class TestMergeShapes:
    def test_basic_merge(self, tmp_path):
        """Shape columns from shape CSV replace N/A in timing CSV."""
        timing_csv = str(tmp_path / "timing.csv")
        shape_csv = str(tmp_path / "shape.csv")
        output_csv = str(tmp_path / "merged.csv")

        _write_csv(
            timing_csv,
            [
                {
                    "Name": "kernel_a",
                    "Duration (us)": "100",
                    "Dims": "N/A",
                    "Data Type": "N/A",
                },
                {
                    "Name": "kernel_b",
                    "Duration (us)": "200",
                    "Dims": "N/A",
                    "Data Type": "N/A",
                },
            ],
        )
        _write_csv(
            shape_csv,
            [
                {
                    "Name": "kernel_a",
                    "Duration (us)": "99",
                    "Dims": "[64, 128]",
                    "Data Type": "float32",
                },
                {
                    "Name": "kernel_b",
                    "Duration (us)": "199",
                    "Dims": "[32, 64]",
                    "Data Type": "bfloat16",
                },
            ],
        )

        result_path = merge_shapes(timing_csv, shape_csv, output_csv)
        assert result_path == output_csv

        rows = _read_csv(output_csv)
        assert len(rows) == 2
        # Timing durations preserved
        assert rows[0]["Duration (us)"] == "100"
        assert rows[1]["Duration (us)"] == "200"
        # Shape columns from shape CSV
        assert rows[0]["Dims"] == "[64, 128]"
        assert rows[0]["Data Type"] == "float32"
        assert rows[1]["Dims"] == "[32, 64]"

    def test_already_has_dims(self, tmp_path):
        """If timing CSV already has dims, keep them."""
        timing_csv = str(tmp_path / "timing.csv")
        shape_csv = str(tmp_path / "shape.csv")
        output_csv = str(tmp_path / "merged.csv")

        _write_csv(
            timing_csv,
            [
                {
                    "Name": "kernel_a",
                    "Duration (us)": "100",
                    "Dims": "[1, 2]",
                    "Data Type": "float32",
                },
            ],
        )
        _write_csv(
            shape_csv,
            [
                {
                    "Name": "kernel_a",
                    "Duration (us)": "99",
                    "Dims": "[64, 128]",
                    "Data Type": "float16",
                },
            ],
        )

        merge_shapes(timing_csv, shape_csv, output_csv)
        rows = _read_csv(output_csv)
        # Original dims preserved
        assert rows[0]["Dims"] == "[1, 2]"
        assert rows[0]["Data Type"] == "float32"

    def test_occurrence_matching(self, tmp_path):
        """Multiple occurrences of same kernel matched by index."""
        timing_csv = str(tmp_path / "timing.csv")
        shape_csv = str(tmp_path / "shape.csv")
        output_csv = str(tmp_path / "merged.csv")

        _write_csv(
            timing_csv,
            [
                {"Name": "matmul", "Duration (us)": "100", "Dims": "N/A"},
                {"Name": "matmul", "Duration (us)": "200", "Dims": "N/A"},
            ],
        )
        _write_csv(
            shape_csv,
            [
                {"Name": "matmul", "Duration (us)": "99", "Dims": "[64, 128]"},
                {
                    "Name": "matmul",
                    "Duration (us)": "199",
                    "Dims": "[256, 512]",
                },
            ],
        )

        merge_shapes(timing_csv, shape_csv, output_csv)
        rows = _read_csv(output_csv)
        assert rows[0]["Dims"] == "[64, 128]"
        assert rows[1]["Dims"] == "[256, 512]"

    def test_unmatched_kernels(self, tmp_path):
        """Unmatched kernels keep N/A dims."""
        timing_csv = str(tmp_path / "timing.csv")
        shape_csv = str(tmp_path / "shape.csv")
        output_csv = str(tmp_path / "merged.csv")

        _write_csv(
            timing_csv,
            [
                {"Name": "kernel_x", "Duration (us)": "100", "Dims": "N/A"},
            ],
        )
        _write_csv(
            shape_csv,
            [
                {
                    "Name": "kernel_y",
                    "Duration (us)": "99",
                    "Dims": "[64, 128]",
                },
            ],
        )

        merge_shapes(timing_csv, shape_csv, output_csv)
        rows = _read_csv(output_csv)
        assert rows[0]["Dims"] == "N/A"

    def test_default_output_path(self, tmp_path):
        """When output_csv is None, default naming is used."""
        timing_csv = str(tmp_path / "timing.csv")
        shape_csv = str(tmp_path / "shape.csv")

        _write_csv(
            timing_csv,
            [
                {"Name": "kernel_a", "Duration (us)": "100", "Dims": "N/A"},
            ],
        )
        _write_csv(
            shape_csv,
            [
                {"Name": "kernel_a", "Duration (us)": "99", "Dims": "[1, 2]"},
            ],
        )

        result_path = merge_shapes(timing_csv, shape_csv)
        expected = str(tmp_path / "timing_merged.csv")
        assert result_path == expected
        assert os.path.exists(expected)


# ---------------------------------------------------------------------------
# merge_shapes_dir
# ---------------------------------------------------------------------------


class TestMergeShapesDir:
    def test_dir_merge(self, tmp_path):
        """Directory mode matches CSVs by (rank, stage)."""
        timing_dir = str(tmp_path / "timing")
        shape_dir = str(tmp_path / "shape")
        output_dir = str(tmp_path / "merged")

        _write_csv(
            os.path.join(timing_dir, "1234-TP-0-DECODE.trace.csv"),
            [{"Name": "kernel_a", "Duration (us)": "100", "Dims": "N/A"}],
        )
        _write_csv(
            os.path.join(shape_dir, "5678-TP-0-DECODE.trace.csv"),
            [{"Name": "kernel_a", "Duration (us)": "99", "Dims": "[1, 2]"}],
        )

        results = merge_shapes_dir(
            timing_dir, shape_dir, output_dir, verbose=False
        )
        assert len(results) == 1
        rows = _read_csv(results[0])
        assert rows[0]["Dims"] == "[1, 2]"
        assert rows[0]["Duration (us)"] == "100"

    def test_dir_stage_filter(self, tmp_path):
        """Stage filter skips non-matching CSVs."""
        timing_dir = str(tmp_path / "timing")
        shape_dir = str(tmp_path / "shape")

        _write_csv(
            os.path.join(timing_dir, "1234-TP-0-DECODE.trace.csv"),
            [{"Name": "kernel_a", "Duration (us)": "100", "Dims": "N/A"}],
        )
        _write_csv(
            os.path.join(timing_dir, "1234-TP-0-EXTEND.trace.csv"),
            [{"Name": "kernel_b", "Duration (us)": "200", "Dims": "N/A"}],
        )
        _write_csv(
            os.path.join(shape_dir, "5678-TP-0-DECODE.trace.csv"),
            [{"Name": "kernel_a", "Duration (us)": "99", "Dims": "[1, 2]"}],
        )
        _write_csv(
            os.path.join(shape_dir, "5678-TP-0-EXTEND.trace.csv"),
            [{"Name": "kernel_b", "Duration (us)": "199", "Dims": "[3, 4]"}],
        )

        results = merge_shapes_dir(
            timing_dir, shape_dir, stage="DECODE", verbose=False
        )
        assert len(results) == 1

    def test_dir_no_matches(self, tmp_path):
        """No matching shape CSVs produces empty results."""
        timing_dir = str(tmp_path / "timing")
        shape_dir = str(tmp_path / "shape")
        os.makedirs(timing_dir)
        os.makedirs(shape_dir)

        _write_csv(
            os.path.join(timing_dir, "1234-TP-0-DECODE.trace.csv"),
            [{"Name": "kernel_a", "Duration (us)": "100", "Dims": "N/A"}],
        )
        # Shape dir has a different rank
        _write_csv(
            os.path.join(shape_dir, "5678-TP-1-DECODE.trace.csv"),
            [{"Name": "kernel_a", "Duration (us)": "99", "Dims": "[1, 2]"}],
        )

        results = merge_shapes_dir(timing_dir, shape_dir, verbose=False)
        assert len(results) == 0
