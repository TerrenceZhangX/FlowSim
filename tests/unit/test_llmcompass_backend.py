import os
import time
import json
from pathlib import Path

import pytest
import requests

from simulator.base_parser import BaseKernelInfoParser
from simulator.utils import parse_kernel_entry
from backend.interface import (
    get_default_api_url,
    submit_task,
    get_result,
    wait_for_health,
    run_init_server,
)

# Use same artifact dir env as other tests
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))


def _ensure_artifacts_dir():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.integration
def test_post_parsed_kernels_to_backend():
    """Parse a real trace and post each parsed kernel to the backend `/tasks` endpoint.

    Environment:
      TRACE_PATH - optional path to trace file (default: /flowsim/tests/unit/test_trace.trace.json.gz)
      API_URL - optional backend url (default: http://127.0.0.1:8000)
    """
    _ensure_artifacts_dir()

    trace_path = os.environ.get(
        "TRACE_PATH", "./tests/unit/test_trace.trace.json.gz"
    )
    run_init_server()  # ensure server is running
    api_url = os.environ.get("API_URL", get_default_api_url()).rstrip("/")

    assert Path(trace_path).exists(), f"Trace file not found: {trace_path}"

    parser = BaseKernelInfoParser(trace_path, enable_comm_calibration=False)
    entries = getattr(parser, "individual_info", None) or []
    assert isinstance(entries, list)
    assert len(entries) > 0

    session = requests.Session()

    submitted = []  # list of dicts: {task_id, out_file, payload}

    # ensure backend is healthy before submitting tasks (best-effort)
    wait_for_health(api_url, timeout=10.0)

    for idx, entry in enumerate(entries, start=1):
        # limit to first 100 entries for test
        if idx > 100:
            break
        # parse the entry using a helper that normalizes 'N/A' values
        kernel_name, input_dim, dtype, op = parse_kernel_entry(entry)

        payload = {
            "kernel_name": kernel_name,
            "op": op,
            "input_dim": input_dim,
            "dtype": dtype,
            "system_key": "A100_4_fp161",
        }

        out_file = ARTIFACT_DIR / f"task_{idx}_{kernel_name[0:10]}.json"

        # submit via helper
        resp = submit_task(api_url, payload, timeout=10, session=session)
        # persist response for debugging
        with open(out_file, "w") as f:
            json.dump({"request": payload, "response": resp}, f, indent=2)

        assert "error" not in resp, f"submit_task error: {resp.get('error')}"
        assert resp.get("status_code") == 200
        body = resp.get("body")
        assert isinstance(body, dict)
        task_id = body.get("task_id")
        assert task_id
        submitted.append(
            {"task_id": task_id, "out_file": out_file, "payload": payload}
        )

        # avoid hammering the server
        time.sleep(0.02)

    # after submitting all tasks, poll them together until all reach terminal state or timeout
    if submitted:
        pending = {s["task_id"]: s for s in submitted}
        poll_deadline = time.time() + max(120.0, len(pending) * 5)
        while time.time() < poll_deadline and pending:
            for task_id in list(pending.keys()):
                res = get_result(api_url, task_id, timeout=10, session=session)

                # write poll artifact for this task
                with open(
                    ARTIFACT_DIR / f"task_{task_id}_poll.json", "w"
                ) as pf:
                    json.dump(res, pf, indent=2)

                if "error" in res:
                    # treat network error as transient; will retry until deadline
                    continue

                if res.get("status") == "done":
                    result = res.get("result")
                    assert isinstance(
                        result, dict
                    ), f"done but no result for {task_id}: {res}"
                    assert result.get("status") in ("success", "failed")
                    pending.pop(task_id, None)

            if pending:
                time.sleep(0.5)

        # after polling loop, ensure no pending tasks remain
        assert (
            not pending
        ), f"some tasks did not reach terminal state: {list(pending.keys())}"
