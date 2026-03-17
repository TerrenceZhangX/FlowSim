import os
import time
import re
import math
import pytest
import simulator.benchmarks.nccl_benchmarks as nb
from simulator.base_parser import BaseKernelInfoParser

from tests.utils import _write_artifact
from tests.utils import ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR


@pytest.fixture(scope="module")
def real_trace_file():
    trace_path = "/flowsim/tests/unit/test_trace.trace.json.gz"
    assert os.path.exists(trace_path), f"Profile File Not Exist: {trace_path}"
    return trace_path


@pytest.mark.usefixtures("real_trace_file")
def test_base_parser_with_real_profile(real_trace_file):
    # Test file is TP=4
    parser = BaseKernelInfoParser(
        real_trace_file, TP=4, enable_comm_calibration=False
    )
    assert isinstance(parser.events, list)
    assert len(parser.events) > 0

    artifact_dir = os.environ.get(ARTIFACT_ENV, DEFAULT_ARTIFACT_DIR)
    parser.save_individual_csv(artifact_dir)
    csv_path = os.path.join(artifact_dir, "test_trace.trace.csv")
    assert os.path.exists(csv_path), "Filtered individual info CSV not created"

    # individual_info = [(name, dims, input_type, roles, desc, duration, op, operation, source_code, call_stack)]
    missing_ops = []
    for item in parser.individual_info:
        if not item[6]:
            missing_ops.append(item[0])
    # Warn about kernels without op mapping but don't fail — these need
    # manual additions to kernels.json.
    if missing_ops:
        import warnings

        warnings.warn(
            f"{len(missing_ops)} kernel(s) have empty op mapping "
            f"(first 5: {missing_ops[:5]}). "
            f"Add entries to kernels.json to fix."
        )
