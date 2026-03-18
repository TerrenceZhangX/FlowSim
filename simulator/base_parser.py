import gzip
import json
import csv
import os

from collections import defaultdict
import re
import simulator.benchmarks.nccl_benchmarks as nb


class BaseKernelInfoParser:
    """
    Parses kernel trace files and extracts detailed information for each kernel event.

    This class processes profiling trace files to retrieve kernel execution details,
    including names, input dimensions, data types, durations, and call stack info.
    It supports mapping external identifiers to kernel metadata and aggregates kernel
    statistics for further analysis.

    Args:
        file_path (str): Path to the gzipped JSON trace file containing event data.

    Attributes:
        file_path (str): The input trace file path.
        base_filename (str): The base name of the input file.
        base_name (str): The base name without extensions.
        events (list): List of parsed trace events.
        individual_info (list): List of tuples with kernel event details.
        total_duration (int): Total duration of all kernel events.

    Returns:
        individual_info (list): A list of tuples, each containing:
            - name (str): Kernel name.
            - dims (any): Input dimensions.
            - input_type (any): Data type of input.
            - roles (str): Placeholder for input/output roles.
            - desc (str): Placeholder for descriptions.
            - duration (int): Duration of the kernel event (us).
            - op (str): Placeholder for operation.
            - operation (str): Placeholder for operation details.
            - source_code (str): Placeholder for source code reference.
            - call_stack (str): Human-readable call stack trace.

    Example:
        [
            ('aten::matmul', [[64, 128], [128, 256]], ['float32', 'float32'],
             '', '', 123, '', '', '', 'LaunchKernel <- forward <- Main'),
            ...
        ]
    """

    def __init__(
        self, file_path: str, TP: int = 8, enable_comm_calibration: bool = True
    ) -> None:
        """Initializes the BaseKernelInfoParser object.

        Loads and parses kernel trace events from the specified gzipped JSON file,
        preparing internal structures for kernel profiling analysis.

        Args:
            file_path (str): Path to the gzipped JSON trace file containing kernel events.

        Attributes:
            file_path (str): Stores the input file path.
            base_filename (str): The base name of the input file.
            base_name (str): The base name without extensions.
            events (list): List of parsed trace events.
            individual_info (list): List of tuples containing kernel event details.
            total_duration (int): Total duration of all kernel events.

        Returns:
            None. Initializes internal state and populates event data for further
            analysis.
        """
        self.file_path = file_path
        self.base_filename = os.path.basename(file_path)
        self.base_name = os.path.splitext(
            os.path.splitext(self.base_filename)[0]
        )[0]

        self.events = []
        # individual_info = [(name, dims, input_type, roles, desc, duration, op, operation, source_code, call_stack)]
        self.individual_info = []
        self.aggregate_kernel_info = []
        self.total_duration = 0
        self.tensor_parallelism = (
            TP  # Number of GPUs in the system for Tensor Parallelism
        )

        self._load_events()
        self._parse_events()

        # By default calibrate communication kernels using NCCL benchmarks
        # User can opt to disable communication time incase it's ignored in simulation
        if enable_comm_calibration:
            self._calibrate_communication_kernels()

        # Add annotations from kernel database
        self.post_process_with_db(db_path="/flowsim/kernels.json")

    def _load_events(self) -> None:
        """
        Loads and parses kernel trace events from a gzipped JSON file.

        Args:
            None. Uses self.file_path as the source file path.

        Returns:
            None. Populates self.events with a list of parsed event dictionaries.
            If loading fails, self.events will be an empty list.
        """
        try:
            with gzip.open(self.file_path, "rt", encoding="utf-8") as f:
                full_data = json.load(f)
            self.events = full_data.get("traceEvents", [])
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            self.events = []

    def _get_callstack(self) -> dict[int, str]:
        """
        Builds a mapping from external or correlation IDs to call stack traces
        for kernel launch events.

        Processes the trace events in self.events, grouping them by process and
        thread IDs, and simulates call stack nesting using event timestamps and
        durations. For each event with an "External id" or "correlation"
        argument and a call stack containing "LaunchKernel", associates the ID
        with the corresponding call stack trace.

        Args:
            None. Uses self.events (list of event dicts) as input.

        Returns:
            A dict mapping external IDs or correlation IDs (as found in event
            arguments) to a string representing the call stack trace at the time
            of kernel launch. The call stack trace is formatted as
            "stack top <- ... <- stack base".

            Example:
            {
                12345: "LaunchKernel <- forward <- Main"
            }
        """
        threads = defaultdict(list)
        extid_to_stack = {}
        for evt in self.events:
            # The complete event is marked with "ph": "X"
            # Incomplete events (like "B" and "E") are not happening in the trace
            if evt.get("ph") == "X":
                pid = evt.get("pid")
                tid = evt.get("tid")
                threads[(pid, tid)].append(evt)

        for (pid, tid), evts in threads.items():
            # Sort events by timestamp for proper nesting
            evts.sort(key=lambda e: e["ts"])

            call_stack = []
            for evt in evts:
                start = evt["ts"]
                dur = evt.get("dur", 0)
                end = start + dur
                name = evt["name"]
                # By design, external_id and correlation_id are mutually exclusive
                external_id = evt.get("args", {}).get("External id")
                correlation_id = evt.get("args", {}).get("correlation")

                # Pop finished frames: ensure the current call_stack only contains
                # events that are still active at 'start'. Each stack entry stores
                # (name, end_time, external_id, correlation_id). If the top frame's
                # end time is <= current event's start, that frame finished before
                # this event began, so remove it.
                while call_stack and call_stack[-1][1] <= start:
                    call_stack.pop()

                # Push current event onto the simulated call stack. We store end time
                # and any ids so we can later map kernels back to annotations.
                call_stack.append((name, end, external_id, correlation_id))

                # Build human readable call stack string from outermost to innermost
                stack_trace = " <- ".join(
                    name for name, _, _, _ in reversed(call_stack)
                )

                # Only record mappings for events that have an External id or a
                # correlation id and where the stack contains a LaunchKernel entry.
                # Reason: we want to associate user-facing ids (External id / correlation)
                # with the higher-level call context that launched GPU kernels.
                if (external_id is not None or correlation_id is not None) and (
                    "LaunchKernel" in stack_trace
                    or "LaunchCooperativeKernel" in stack_trace
                ):
                    if external_id is not None:
                        extid_to_stack[external_id] = stack_trace
                    elif correlation_id is not None:
                        extid_to_stack[correlation_id] = stack_trace
                    else:
                        # This should never happen, otherwise the kernel is obsolete
                        print(
                            f"Error: Event {name} at {start}us has no External id or correlation id."
                        )

        return extid_to_stack

    def _parse_events(self) -> list[tuple]:
        """
        Parses kernel trace events and extracts structured kernel execution details.

        Processes self.events, a list of event dictionaries from a profiler trace,
        to build a summary of GPU kernel executions. For each kernel event, this
        method maps external IDs to input dimensions and types, matches kernel
        events with user/system annotations, and retrieves the call stack for each
        kernel. The extracted information is stored in self.individual_info as a
        list of tuples, each containing:

            (name, dims, input_type, roles, desc, duration, op, operation,
             source_code, call_stack)

        Args:
            None. Operates on self.events, which should be populated with profiling
            event data.

        Returns:
            list: A list of tuples, each representing a kernel event with the
            following fields:
            - name (str): Kernel name.
            - dims (any): Input dimensions.
            - input_type (any): Data type of input.
            - roles (str): Placeholder for input/output roles.
            - desc (str): Placeholder for descriptions.
            - duration (int): Duration of the kernel event in microseconds.
            - op (str): Placeholder for operation.
            - operation (str): Placeholder for operation details.
            - source_code (str): Placeholder for source code reference.
            - call_stack (str): Human-readable call stack trace.

        Example:
            [
            ('aten::matmul', [[64, 128], [128, 256]],
             ['float32', 'float32'], '', '', 123, '', '', '',
             'LaunchKernel <- forward <- Main'),
            ...
            ]
        """
        extid_to_dims = {}
        extid_to_type = {}
        self.individual_info = []
        annotation_events = [
            entry
            for entry in self.events
            if (
                entry.get("cat") == "gpu_user_annotation"
                or entry.get("cat") == "user_annotation"
            )
        ]

        # Map External id to Input Dimensions and Input type
        for entry in self.events:
            args = entry.get("args", {})
            ext_id = args.get("External id")
            dims = args.get("Input Dims")
            input_type = args.get("Input type")
            if ext_id is not None and dims is not None:
                extid_to_dims[ext_id] = dims
                extid_to_type[ext_id] = input_type

        # Get call stack for each kernel using External id or correlation id
        kernel_call_stack = self._get_callstack()
        query_name_counter = defaultdict(int)

        # Enforce one-to-one matching between annotation events and kernels
        # (each annotation can be consumed at most once).
        used_annotations: set[int] = set()

        for entry in self.events:
            if entry.get("cat") == "kernel":
                # Kernel event intermediate parameters
                args = entry.get("args", {})
                ext_id = args.get("External id")
                correlation_id = args.get("correlation")
                start = entry.get("ts", 0)
                end = start + entry.get("dur", 0)
                # Kernel information to be recorded
                # Format: [(name, dims, input_type, roles, desc, duration, op, operation, source_code, call_stack)]
                name = entry.get("name")
                dims = extid_to_dims.get(ext_id, "N/A")
                input_type = extid_to_type.get(ext_id, "N/A")
                duration = entry.get("dur", 0)
                # roles, desc, op, operation, source_code are added from database later
                idx = query_name_counter[name]
                query_name_counter[name] += 1
                if ext_id is not None:
                    call_stack = kernel_call_stack.get(ext_id)
                elif correlation_id is not None:
                    call_stack = kernel_call_stack.get(correlation_id)
                else:
                    call_stack = None
                    print(
                        f"Warning: kernel {name} at {start}us has no upper stream External id or correlation id."
                    )

                # Case 1: Torch profiler linked external id with input dims and type
                if ext_id is not None and dims != "N/A":
                    self.individual_info.append(
                        (
                            name,
                            dims,
                            input_type,
                            "",
                            "",
                            duration,
                            "",
                            "",
                            "",
                            call_stack,
                        )
                    )
                else:
                    # Case 2: If no ext_id, we need to find the shape from user annotations
                    # Key Identification Methodology: Annotation is overlapped with kernel
                    dims_anno = "N/A"
                    input_type_anno = "N/A"
                    desc_anno = ""
                    for anno_idx, anno in enumerate(annotation_events):
                        if anno_idx in used_annotations:
                            continue
                        if "ProfilerStep" in anno.get("name", ""):
                            continue
                        anno_start = anno.get("ts", 0)
                        anno_end = anno_start + anno.get("dur", 0)
                        if "nccl" in name.lower():
                            buffer = 1000  # 1ms buffer for NCCL annotations due to launch delay
                        else:
                            buffer = 10  # 10us buffer for almost overlapping annotations
                        # Check if the kernel's time range overlaps with the annotation's time range
                        if anno_start - buffer <= start <= anno_end + buffer:
                            name_anno = anno.get("name", "")

                            # If we have call stack context, require annotation prefix to match
                            # something in the stack (time + name), to avoid accidental matches.
                            if call_stack and "|" in name_anno:
                                prefix_anno = name_anno.split("|", 1)[0].strip()
                                if (
                                    prefix_anno
                                    and prefix_anno not in call_stack
                                ):
                                    continue

                            if "nccl" in name.lower():
                                # Avoid nccl kernel matching other annotations
                                if not (
                                    "nccl" in name_anno
                                    or "attn_tp_reduce_scatter" in name_anno
                                ):
                                    continue

                            parsed_dims, parsed_types, parsed_names = (
                                self._parse_dims_and_types_from_annotation_name(
                                    name_anno
                                )
                            )

                            # Filter: if name-based parsing yields no dims, ignore this annotation.
                            if not parsed_dims:
                                used_annotations.add(anno_idx)
                                continue

                            # print(f"Parsed dims from annotation: {parsed_dims}, name annotation: {name_anno}, kernel name: {parsed_names}")
                            dims_anno = parsed_dims
                            input_type_anno = parsed_types
                            desc_anno = parsed_names
                            break
                    self.individual_info.append(
                        (
                            name,
                            dims_anno,
                            input_type_anno,
                            "",
                            desc_anno,
                            duration,
                            "",
                            "",
                            "",
                            call_stack,
                        )
                    )

        return self.individual_info

    @staticmethod
    def _parse_dims_and_types_from_annotation_name(
        annotation_name: str,
    ) -> tuple[list[list[int]], list[str], list[str]]:
        """Parse dims/dtypes/variable-names from an annotation event name.

        Supports:
        - New format: `kernel_name|var0[4x16:float32],var1[4x8:float32]`
        - Legacy format: embedded `dtype=[...]` and `x=(...)` style segments.

        Returns empty lists when parsing fails.
        """

        if not annotation_name:
            return [], [], []

        # New format: prefix `kernel|`, then comma-separated `var[AxBx...:dtype]`
        if "|" not in annotation_name:
            return [], [], []

        _, payload = annotation_name.split("|", 1)
        dims: list[list[int]] = []
        dtypes: list[str] = []
        var_names: list[str] = []
        for token in (t.strip() for t in payload.split(",")):
            if not token:
                continue
            m = re.match(
                r"(?P<var>[^\[]+)\[(?P<shape>[^:\]]+):(?P<dtype>[^\]]+)\]$",
                token,
            )
            if not m:
                continue
            var_name = m.group("var").strip()
            shape_str = m.group("shape").strip()
            dtype_str = m.group("dtype").strip()
            if not shape_str:
                continue
            try:
                shape = [
                    int(p)
                    for p in re.split(r"[xX]", shape_str)
                    if p.strip() != ""
                ]
            except ValueError:
                continue
            if shape:
                dims.append(shape)
                dtypes.append(dtype_str)
                var_names.append(var_name)
        if dims:
            return dims, dtypes, var_names

        return [], [], []

    def _calibrate_communication_kernels(self) -> None:
        """
        Calibrates the durations of communication kernels (e.g., all_reduce, all_gather)
        in self.individual_info using NCCL benchmarks and replaces the sampled durations.

        Args:
            None. Operates on and updates self.individual_info and uses
            self.tensor_parallelism to parameterize NCCL benchmarking.

        Returns:
            None.

        Notes:
            - [Important] This function must be called after _parse_events()
            - Durations in self.individual_info are updated in-place.
            - A local cache of profiled durations is used to reduce repeated benchmarking.
        """
        pytorch_to_nccl_dtype = {
            "c10::BFloat16": "bfloat16",
            "torch.bfloat16": "bfloat16",
            "TensorList": "bfloat16",  # Assuming TensorList is treated as bfloat16 for NCCL
        }
        pytorch_to_nccl_byte = {
            "c10::BFloat16": 2,
            "torch.bfloat16": 2,
            "TensorList": 2,
        }

        # Cache to store profiled durations for (name, dtype, size)
        comm_profile_cache = {}

        for i, (
            name,
            dims,
            input_type,
            roles,
            desc,
            duration,
            op,
            operation,
            source_code,
            call_stack,
        ) in enumerate(self.individual_info):
            stack_parts = [p.strip() for p in call_stack.split("<-")]
            if len(stack_parts) > 1:
                kernel_impl = stack_parts[1]
            else:
                kernel_impl = stack_parts[0] if stack_parts else ""

            if kernel_impl == "nccl:all_reduce":
                # nccl's all_reduce kernel
                shape = dims[0][0]
                dtype = input_type[0]
                nccl_dtype = pytorch_to_nccl_dtype.get(dtype)
                byte_size = pytorch_to_nccl_byte.get(dtype)
                if nccl_dtype is None or byte_size is None:
                    continue
                size = shape[0] * shape[1] * byte_size
                cache_key = (name, dtype, size)
                if cache_key in comm_profile_cache:
                    profiled_duration = comm_profile_cache[cache_key]
                else:
                    # Parameters:
                    # -b: min bytes
                    # -e: max bytes
                    # -g: number of GPUs
                    # -d: data type
                    profiled_duration = nb.run_nccl_all_reduce_perf(
                        cmd_path="/flowsim/third_party/nccl-tests/build/all_reduce_perf",
                        b=str(size),
                        e=str(size),
                        g=str(self.tensor_parallelism),
                        d=nccl_dtype,
                    )
                    comm_profile_cache[cache_key] = profiled_duration
                if profiled_duration is None:
                    continue
                self.individual_info[i] = (
                    name,
                    dims,
                    input_type,
                    roles,
                    desc,
                    profiled_duration,
                    op,
                    operation,
                    source_code,
                    call_stack,
                )
            elif kernel_impl == "sgl_kernel::all_reduce":
                # Sglang's custom all_reduce kernel
                shape = dims[1]
                dtype = input_type[1]
                nccl_dtype = pytorch_to_nccl_dtype.get(dtype)
                byte_size = pytorch_to_nccl_byte.get(dtype)
                if nccl_dtype is None or byte_size is None:
                    continue
                size = shape[0] * shape[1] * byte_size
                cache_key = (name, dtype, size)
                if cache_key in comm_profile_cache:
                    profiled_duration = comm_profile_cache[cache_key]
                else:
                    profiled_duration = nb.run_nccl_all_reduce_perf(
                        cmd_path="/flowsim/third_party/nccl-tests/build/all_reduce_perf",
                        b=str(size),
                        e=str(size),
                        g=str(self.tensor_parallelism),
                        d=nccl_dtype,
                    )
                    comm_profile_cache[cache_key] = profiled_duration
                if profiled_duration is None:
                    continue
                self.individual_info[i] = (
                    name,
                    dims,
                    input_type,
                    roles,
                    desc,
                    profiled_duration,
                    op,
                    operation,
                    source_code,
                    call_stack,
                )
            elif kernel_impl == "nccl:_all_gather_base":
                # nccl's all_gather kernel
                shape = dims[0]
                dtype = input_type[0]
                nccl_dtype = pytorch_to_nccl_dtype.get(dtype)
                byte_size = pytorch_to_nccl_byte.get(dtype)
                if nccl_dtype is None or byte_size is None:
                    continue
                size = shape[0] * shape[1] * byte_size
                cache_key = (name, dtype, size)
                if cache_key in comm_profile_cache:
                    profiled_duration = comm_profile_cache[cache_key]
                else:
                    profiled_duration = nb.run_nccl_all_gather_perf(
                        cmd_path="/flowsim/third_party/nccl-tests/build/all_gather_perf",
                        b=str(size),
                        e=str(size),
                        g=str(self.tensor_parallelism),
                        d=nccl_dtype,
                    )
                    comm_profile_cache[cache_key] = profiled_duration
                if profiled_duration is None:
                    continue
                self.individual_info[i] = (
                    name,
                    dims,
                    input_type,
                    roles,
                    desc,
                    profiled_duration,
                    op,
                    operation,
                    source_code,
                    call_stack,
                )
            elif op == "all_reduce" or op == "all_gather":
                print(f"Unsupported communication kernel in benchmark: {name}")
            else:
                # Skip non-communication kernels
                continue

    def post_process_with_db(
        self, db_path: str = "/flowsim/kernels.json"
    ) -> None:
        """
        Post-process the individual kernel info with the kernel database.
        Post-prcessing logic:
        1. Load the kernel database from the specified JSON file.
        2. For each kernel in self.individual_info, attempt to find a matching entry
           in the database using either kernel_name or kernel_implementation.
        3. If a match is found, update the kernel's fields with information from the database,
        4. If no match is found, add the kernel to unknown_kernels.json and ask user to update.

        Arguments:
            db_path (str): Path to the kernel database JSON file to use for enrichment. Defaults to '/flowsim/kernels.json'.
        Returns:
            None. Modifies self.individual_info in-place and may create/update '/flowsim/unknown_kernels.json'.

        Example Database Entry:
        {
            "kernel_name": "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
            "kernel_implementation": "aten::mm",
            "op_mapping": "matmul",
            "operation": "out = mat1 @ mat2",
            "source_code": "mm(Tensor self, Tensor mat2) -> Tensor",
            "call_stack": "cudaLaunchKernelExC <- aten::mm <- aten::matmul <- <built-in method matmul of type object at 0x7352e621fec0> <- sglang/srt/layers/logits_processor.py(435): _get_logits <- sglang/srt/layers/logits_processor.py(242): forward <- torch/nn/modules/module.py(1743): _call_impl <- nn.Module: LogitsProcessor_0 <- sglang/srt/models/gpt2.py(249): forward <- sglang/srt/model_executor/model_runner.py(1151): forward_extend <- sglang/srt/model_executor/model_runner.py(1208): _forward_raw <- sglang/srt/model_executor/model_runner.py(1187): forward <- sglang/bench_one_batch.py(233): extend <- torch/utils/_contextlib.py(113): decorate_context <- sglang/bench_one_batch.py(389): latency_test_run_once <- sglang/bench_one_batch.py(499): latency_test <- sglang/bench_one_batch.py(548): main <- sglang/bench_one_batch.py(589): <module> <- runpy.py(86): _run_code <- runpy.py(196): _run_module_as_main",
            "params": [
                {
                    "id": 0,
                    "role": "input",
                    "example_dim": [1, 12288],
                    "example_dtype": "c10::BFloat16",
                    "description": "mat1"
                },
                {
                    "id": 1,
                    "role": "input",
                    "example_dim": [12288, 6288],
                    "example_dtype": "c10::BFloat16",
                    "description": "mat2"
                }
            ]
        }

        Notes:
        1. Role: Input/Ouput, Annotates the I/O relationship of each variable in dimension
        2. Description: What is the corresponding variable of each dimension
        3. operation: The mathematical operation of a kernel
        4. Source Code: Letting developers to find the kernel implemntation easily
        5. Maintaince of Kernel Database is purely manual.
        Per running the parser, it will generate an unknown_kernel.json if the kernel is not seen.
        To insert a db entry, one needs to copy the entry from unknown_kernel.json into kernels.json
        and add information
        """
        if not os.path.exists(db_path):
            print(f"Database file {db_path} does not exist.")
            db_data = {}
        else:
            # Load the database
            with open(db_path, "r") as db_file:
                db_data = json.load(db_file)

            if isinstance(db_data, list):
                db_data_kernel_name = {
                    item["kernel_name"]: item
                    for item in db_data
                    if "kernel_name" in item
                }
                db_data_kernel_impl = {
                    item["kernel_implementation"]: item
                    for item in db_data
                    if "kernel_implementation" in item
                }

        # Default empty maps when db is missing or not a list.
        db_data_kernel_name = locals().get("db_data_kernel_name", {})
        db_data_kernel_impl = locals().get("db_data_kernel_impl", {})

        unknown_path = "/flowsim/unknown_kernels.json"
        unknown_list = []
        if os.path.exists(unknown_path):
            with open(unknown_path, "r") as f:
                try:
                    unknown_list = json.load(f)
                except Exception:
                    unknown_list = []

        # Update individual info with the database
        for i, (
            name,
            dims,
            input_type,
            roles,
            desc,
            duration,
            op,
            operation,
            source_code,
            call_stack,
        ) in enumerate(self.individual_info):
            # Extract op from call stack if not provided
            # Example stack: cudaLaunchKernel <- aten::cumsum <- <built-in method cumsum of type object at 0x75145c4f6f40> <- ....
            kernel_impl = ""
            if call_stack is not None:
                stack_parts = [p.strip() for p in call_stack.split("<-")]
                if len(stack_parts) > 1:
                    kernel_impl = stack_parts[1]
                else:
                    kernel_impl = stack_parts[0] if stack_parts else ""

            db_entry = None
            if name in db_data_kernel_name:
                db_entry = db_data_kernel_name[name]
            elif (
                kernel_impl
                and kernel_impl in db_data_kernel_impl
                and "<built-in function launch>" not in kernel_impl
                and "cuLaunchKernelEx" not in kernel_impl
            ):
                db_entry = db_data_kernel_impl[kernel_impl]

            if db_entry:
                valid_params = [p for p in db_entry.get("params", [])]

                op_mapping = db_entry.get("op_mapping", "")
                # Get the descriptions for the inputs
                kernel_param_desc = [
                    p.get("description", "") for p in valid_params
                ]
                # Get input/output type from database, under params - role
                roles = [p.get("role", "") for p in valid_params]
                # Point to the source code of the kernel
                source_code = db_entry.get("source_code", "")
                # Get the mathmetical operation from database
                operation = db_entry.get("operation", "")

                # Special Handling for gemm kernels where bias could not included
                if (
                    op_mapping == "matmul"
                    and isinstance(dims, (list, tuple))
                    and len(dims) == 2
                ):
                    # If dims is a 2D list, it means it's a matrix multiplication without bias
                    roles = ["input", "input"]
                    kernel_param_desc = ["A", "B"]
                    operation = "C = A @ B"

                self.individual_info[i] = (
                    name,
                    dims,
                    input_type,
                    roles,
                    kernel_param_desc,
                    duration,
                    op_mapping,
                    operation,
                    source_code,
                    call_stack,
                )
            # Not in the database, need to add to unknown kernels
            else:
                # Keep same parameters as before
                dims_list = dims if isinstance(dims, (list, tuple)) else [dims]
                types_list = (
                    input_type
                    if isinstance(input_type, (list, tuple))
                    else [input_type]
                )
                params = []
                for idx in range(max(len(dims_list), len(types_list))):
                    dim = dims_list[idx] if idx < len(dims_list) else []
                    dtype = types_list[idx] if idx < len(types_list) else ""
                    params.append(
                        {
                            "id": idx,
                            "role": "unknown",
                            "example_dim": dim,
                            "example_dtype": dtype,
                            "description": (
                                desc[idx]
                                if isinstance(desc, (list, tuple))
                                and idx < len(desc)
                                else ""
                            ),
                        }
                    )
                unknown_kernel = {
                    "kernel_name": name,
                    "kernel_implementation": kernel_impl,
                    "op_mapping": "",
                    "operation": "",
                    "source_code": "",
                    "call_stack": call_stack,
                    "params": params,
                }
                # Update the individual info with the unknown kernel
                self.individual_info[i] = (
                    name,
                    dims,
                    input_type,
                    "",
                    desc,
                    duration,
                    "",
                    "",
                    "",
                    call_stack,
                )
                # Avoid duplicates
                if not any(
                    item.get("kernel_name") == name for item in unknown_list
                ):
                    unknown_list.append(unknown_kernel)
                    with open(unknown_path, "w") as f:
                        json.dump(unknown_list, f, ensure_ascii=False, indent=2)

        with open(unknown_path, "w") as f:
            json.dump(unknown_list, f, ensure_ascii=False, indent=2)

    def get_aggregate_kernel_info(self) -> list[tuple]:
        """
        Aggregates kernel profiling data by grouping entries with identical name,
        dimensions, and input type.

        Returns:
            list: A list mapping each unique (name, dims, input_type) combination
            to its aggregated profiling data.
        """
        # Create a dictionary to store folded information
        folded_info = {}
        for (
            name,
            dims,
            input_type,
            io,
            desc,
            duration,
            op,
            operation,
            source_code,
            call_stack,
        ) in self.individual_info:
            key = (name, str(dims), str(input_type))
            if key in folded_info:
                folded_info[key][0] += duration
                folded_info[key][1] += 1
            else:
                folded_info[key] = [duration, 1]

        # Convert back to list format with count
        self.aggregate_kernel_info = [
            (name, dims, input_type, duration, count)
            for (name, dims, input_type), (
                duration,
                count,
            ) in folded_info.items()
        ]
        return self.aggregate_kernel_info

    def get_kernel_e2e_time(self, individual_info: list[tuple]) -> float:
        """
        Calculates the total end-to-end (E2E) duration for the provided
        kernel events.

        Args:
            individual_info (list): A list of tuples, each representing a
            kernel event. The duration for each event is expected at index
            5 of the tuple.

        Returns:
            float: The total E2E duration (in microseconds) for all kernel
            events in `individual_info`.
        """
        kernel_e2e_time = 0
        for kernel in individual_info:
            real_duration = kernel[5]  # Duration in microseconds
            kernel_e2e_time += real_duration

        return kernel_e2e_time

    def save_individual_csv(self, output_dir: str = ".") -> None:
        """
        Writes the contents of `self.individual_info` to a CSV file in the specified
        output directory.

        Args:
            output_dir (str): Directory where the CSV file will be saved. Defaults to
            the current directory.

        Returns:
            None. The CSV file is created at the specified location, containing one
            row per kernel event.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, f"{self.base_name}.csv")
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
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
            )
            csv_writer.writerows(self.individual_info)
