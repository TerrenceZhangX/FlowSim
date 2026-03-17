import subprocess

# NCCL test support data types
#    "int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"
#    "bfloat16"
#    "f8e4m3", "f8e5m2"


def run_nccl_all_reduce_perf(
    cmd_path: str = "./build/all_reduce_perf",
    b: str = "8",
    e: str = "128M",
    f: str = "2",
    g: str = "8",
    d: str = "float",
) -> float | None:
    """
    Run the NCCL all-reduce performance benchmark and parse the reported metric.

    Args:
        cmd_path (str): Path to the all_reduce_perf binary.
        b (str): Minimum message size to test (bytes or human-friendly, e.g., '8').
        e (str): Maximum message size to test (e.g., '128M').
        f (str): Factor/step parameter passed to -f.
        g (str): Number of GPUs to use for the benchmark.
        d (str): Data type indicator (e.g., 'float').

    Returns:
        float | None: Parsed numeric metric from the first data line of the tool's
                       output (float(fields[5]) in current parsing). Returns None
                       if no parsable data line is found.

    """
    cmd = [cmd_path, "-b", b, "-e", e, "-f", f, "-g", g, "-d", d]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out = result.stdout

    out_of_place_time = None

    # Example output as below. We need to extract the time(us) information as data
    # Collective test starting: all_reduce_perf
    # nThread 1 nGpus 8 minBytes 134217728 maxBytes 134217728 step: 1048576(bytes) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0
    #
    # Using devices
    #  Rank  0 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  0 [0001:00:00] NVIDIA H200
    #  Rank  1 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  1 [0002:00:00] NVIDIA H200
    #  Rank  2 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  2 [0003:00:00] NVIDIA H200
    #  Rank  3 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  3 [0008:00:00] NVIDIA H200
    #  Rank  4 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  4 [0009:00:00] NVIDIA H200
    #  Rank  5 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  5 [000a:00:00] NVIDIA H200
    #  Rank  6 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  6 [000b:00:00] NVIDIA H200
    #  Rank  7 Group  0 Pid  34386 on RIC20PrdGPC030002Z6 device  7 [000c:00:00] NVIDIA H200
    #
    #                                                              out-of-place                       in-place
    #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
    #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
    # 134217728      33554432     float     sum      -1    582.1  230.57  403.50      0    581.7  230.73  403.77      0
    # Out of bounds values : 0 OK
    # Avg bus bandwidth    : 403.638
    #
    # Collective test concluded: all_reduce_perf

    for line in out.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            fields = line.split()
            if len(fields) >= 6:
                try:
                    out_of_place_time = float(fields[5])
                except ValueError:
                    continue
                break

    return out_of_place_time


def run_nccl_all_gather_perf(
    cmd_path: str = "./build/all_gather_perf",
    b: str = "8",
    e: str = "128M",
    f: str = "2",
    g: str = "8",
    d: str = "float",
) -> float | None:
    """
    Run the NCCL all-gather performance benchmark and return the out-of-place times.

    Args:
        cmd_path (str): Path to the all_gather_perf binary.
        b (str): Minimum message size to test (bytes or human-friendly, e.g., '8').
        e (str): Maximum message size to test (e.g., '128M').
        f (str): Factor/step parameter passed to -f.
        g (str): Number of GPUs to use for the benchmark.
        d (str): Data type indicator (e.g., 'float').

    Returns:
        float | None: Parsed numeric metric from the first data line of the tool's
                       output (float(fields[5]) in current parsing). Returns None
                       if no parsable data line is found.
    """
    cmd = [cmd_path, "-b", b, "-e", e, "-f", f, "-g", g, "-d", d]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out = result.stdout

    out_of_place_time = None

    for line in out.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            fields = line.split()
            if len(fields) >= 6:
                try:
                    out_of_place_time = float(fields[5])
                except ValueError:
                    continue
                break

    return out_of_place_time
