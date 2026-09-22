import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_time_v(stderr_text):
    """Parse useful fields from /usr/bin/time -v output."""
    out = {}

    patterns = {
        "time_elapsed_wall": r"Elapsed \(wall clock\) time.*:\s*(.+)",
        "time_user_sec": r"User time \(seconds\):\s*(.+)",
        "time_system_sec": r"System time \(seconds\):\s*(.+)",
        "cpu_percent": r"Percent of CPU this job got:\s*(.+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*(.+)",
        "minor_page_faults": r"Minor \(reclaiming a frame\) page faults:\s*(.+)",
        "major_page_faults": r"Major \(requiring I/O\) page faults:\s*(.+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, stderr_text)
        if m:
            value = m.group(1).strip()
            if key in ["time_user_sec", "time_system_sec"]:
                try:
                    value = float(value)
                except ValueError:
                    pass
            elif key in ["max_rss_kb", "minor_page_faults", "major_page_faults"]:
                try:
                    value = int(value)
                except ValueError:
                    pass
            out[key] = value

    return out


def parse_json_stdout(stdout_text):
    """qtn_baseline.py prints JSON to stdout."""
    text = stdout_text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: find first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def run_one(args, mode, nqubits, log_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{timestamp}_{args.workload}_{args.output}_{mode}_n{nqubits}"

    if args.ngates is not None:
        tag += f"_g{args.ngates}"

    if args.output == "expectation":
        tag += f"_{args.observable}"

    if mode == "mps":
        tag += f"_bond{args.max_bond}"

    if args.threads is not None:
        tag += f"_thr{args.threads}"

    stdout_file = log_dir / f"{tag}.stdout.json"
    stderr_file = log_dir / f"{tag}.stderr.log"

    baseline_cmd = [
        sys.executable,
        args.baseline,
        "--mode",
        mode,
        "--output",
        args.output,
        "--workload",
        args.workload,
        "--nqubits",
        str(nqubits),
        "--optimizer",
        args.optimizer,
        "--observable",
        args.observable,
        "--svd-cutoff",
        str(args.svd_cutoff),
        "--expectation-engine",
        args.expectation_engine,
        "--two-qubit-apply",
        args.two_qubit_apply,
        "--max-statevector-qubits",
        str(args.max_statevector_qubits),
    ]

    if args.ngates is not None:
        baseline_cmd += ["--ngates", str(args.ngates)]

    if args.check_ref and nqubits <= args.check_ref_max_qubits:
        baseline_cmd += [
            "--check-ref",
            "--check-ref-max-qubits",
            str(args.check_ref_max_qubits),
        ]

    if args.profile_conversion:
        baseline_cmd += ["--profile-conversion"]

    if args.fuse_single_qubit:
        baseline_cmd += ["--fuse-single-qubit"]

    if args.absorb_1q_into_2q:
        baseline_cmd += ["--absorb-1q-into-2q"]

    if args.threads is not None:
        baseline_cmd += ["--threads", str(args.threads)]

    if mode == "mps" and args.max_bond is not None:
        baseline_cmd += ["--max-bond", str(args.max_bond)]

    time_v = Path("/usr/bin/time")
    use_time_v = time_v.exists()
    cmd = [
        "/usr/bin/time",
        "-v",
        *baseline_cmd,
    ] if use_time_v else baseline_cmd

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    if args.threads is not None:
        for name in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            env[name] = str(args.threads)

    print("Running:", " ".join(cmd), flush=True)

    wall_t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    runner_wall_sec = time.perf_counter() - wall_t0

    stdout_file.write_text(proc.stdout, encoding="utf-8")
    stderr_file.write_text(proc.stderr, encoding="utf-8")

    row = {
        "timestamp": timestamp,
        "mode": mode,
        "output": args.output,
        "workload": args.workload,
        "nqubits": nqubits,
        "requested_ngates": args.ngates if args.ngates is not None else "",
        "max_bond": args.max_bond if mode == "mps" else "",
        "svd_cutoff": args.svd_cutoff if mode == "mps" else "",
        "observable": args.observable if args.output == "expectation" else "",
        "threads": args.threads if args.threads is not None else "",
        "optimizer": args.optimizer,
        "expectation_engine": args.expectation_engine,
        "two_qubit_apply": args.two_qubit_apply,
        "returncode": proc.returncode,
        "runner_wall_sec": runner_wall_sec,
        "used_usr_bin_time": use_time_v,
        "stdout_log": str(stdout_file),
        "stderr_log": str(stderr_file),
    }

    try:
        meta = parse_json_stdout(proc.stdout)
        row.update(
            {
                "elapsed_sec_internal": meta.get("elapsed_sec"),
                "stage_total_sec": meta.get("stage_total_sec"),
                "stage_set_threads_sec": meta.get("stage_set_threads_sec"),
                "stage_set_backend_sec": meta.get("stage_set_backend_sec"),
                "stage_build_circuit_sec": meta.get("stage_build_circuit_sec"),
                "stage_execute_sec": meta.get("stage_execute_sec"),
                "stage_qibo_to_quimb_sec": meta.get("stage_qibo_to_quimb_sec"),
                "stage_quimb_circuit_init_sec": meta.get(
                    "stage_quimb_circuit_init_sec"
                ),
                "stage_gate_loop_sec": meta.get("stage_gate_loop_sec"),
                "stage_gate_metadata_sec": meta.get("stage_gate_metadata_sec"),
                "stage_gate_apply_sec": meta.get("stage_gate_apply_sec"),
                "stage_gate_apply_1q_sec": meta.get("stage_gate_apply_1q_sec"),
                "stage_gate_apply_2q_sec": meta.get("stage_gate_apply_2q_sec"),
                "stage_gate_apply_other_sec": meta.get("stage_gate_apply_other_sec"),
                "stage_gate_fusion_matrix_sec": meta.get(
                    "stage_gate_fusion_matrix_sec"
                ),
                "stage_gate_absorb_matrix_sec": meta.get(
                    "stage_gate_absorb_matrix_sec"
                ),
                "stage_expectation_contract_sec": meta.get(
                    "stage_expectation_contract_sec"
                ),
                "stage_mps_batched_attempt_sec": meta.get(
                    "stage_mps_batched_attempt_sec"
                ),
                "actual_gate_count": meta.get("actual_gate_count"),
                "gate_count_1q": meta.get("gate_count_1q"),
                "gate_count_2q": meta.get("gate_count_2q"),
                "gate_count_other": meta.get("gate_count_other"),
                "gate_count_measure": meta.get("gate_count_measure"),
                "gate_count_applied_1q": meta.get("gate_count_applied_1q"),
                "gate_count_applied_2q": meta.get("gate_count_applied_2q"),
                "gate_count_applied_other": meta.get("gate_count_applied_other"),
                "gate_count_absorbed_1q": meta.get("gate_count_absorbed_1q"),
                "state_shape": meta.get("state_shape"),
                "state_dtype": meta.get("state_dtype"),
                "state_norm": meta.get("state_norm"),
                "expectation_value": meta.get("expectation_value"),
                "expectation_engine": meta.get(
                    "expectation_engine", args.expectation_engine
                ),
                "two_qubit_apply": meta.get("two_qubit_apply", args.two_qubit_apply),
                "expectation_terms": meta.get("expectation_terms"),
                "expectation_engine_fallback": meta.get(
                    "expectation_engine_fallback"
                ),
                "qibo": meta.get("versions", {}).get("qibo"),
                "qibotn": meta.get("versions", {}).get("qibotn"),
                "qibojit": meta.get("versions", {}).get("qibojit"),
                "quimb": meta.get("versions", {}).get("quimb"),
                "cotengra": meta.get("versions", {}).get("cotengra"),
            }
        )

        ref = meta.get("reference", {})
        row.update(
            {
                "check_ref": bool(ref),
                "ref_elapsed_sec": ref.get("elapsed_sec"),
                "max_abs_error": ref.get("max_abs_error"),
                "allclose_1e-8": ref.get("allclose_1e-8"),
                "expected_expectation_value": ref.get("expected_expectation_value"),
                "abs_error": ref.get("abs_error"),
                "isclose_1e-8": ref.get("isclose_1e-8"),
                "reference_skipped": ref.get("skipped"),
                "reference_skip_reason": ref.get("reason"),
            }
        )
    except Exception as exc:
        row["parse_json_error"] = repr(exc)

    if use_time_v:
        row.update(parse_time_v(proc.stderr))

    if proc.returncode != 0:
        row["status"] = "FAIL"
    elif (
        row.get("check_ref")
        and args.output == "statevector"
        and row.get("allclose_1e-8") is not True
    ):
        row["status"] = "WRONG"
    elif (
        row.get("check_ref")
        and args.output == "expectation"
        and row.get("isclose_1e-8") is not True
    ):
        row["status"] = "WRONG"
    else:
        row["status"] = "OK"

    return row


def append_csv(path, rows):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    exists = path.exists()
    old_fieldnames = []

    if exists and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            old_fieldnames = next(reader)

    if exists and old_fieldnames:
        for key in old_fieldnames:
            if key not in fieldnames:
                fieldnames.insert(0, key)
        for key in fieldnames:
            if key not in old_fieldnames:
                old_fieldnames.append(key)
        fieldnames = old_fieldnames

    write_header = not exists or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="qtn_baseline.py")
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 10, 12, 14, 16])
    parser.add_argument("--modes", nargs="+", choices=["tn", "mps"], default=["tn", "mps"])
    parser.add_argument("--output", choices=["statevector", "expectation"], default="statevector")
    parser.add_argument("--workload", choices=["qft", "random-local"], default="qft")
    parser.add_argument("--ngates", type=int, default=None)
    parser.add_argument("--observable", choices=["z0", "zz-chain"], default="z0")
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--svd-cutoff", type=float, default=1e-10)
    parser.add_argument(
        "--expectation-engine",
        choices=["local", "mps-batched", "auto"],
        default="local",
    )
    parser.add_argument("--two-qubit-apply", choices=["name", "matrix"], default="name")
    parser.add_argument("--profile-conversion", action="store_true")
    parser.add_argument("--fuse-single-qubit", action="store_true")
    parser.add_argument("--absorb-1q-into-2q", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--optimizer", default="auto-hq")
    parser.add_argument("--check-ref", action="store_true")
    parser.add_argument("--check-ref-max-qubits", "--check-ref-max-n", type=int, default=20)
    parser.add_argument("--max-statevector-qubits", type=int, default=30)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    if args.workload == "qft" and args.ngates is not None:
        parser.error("--ngates is only valid with --workload random-local")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = log_dir / "summary.csv"
    summary_jsonl = log_dir / "summary.jsonl"

    all_rows = []

    for nqubits in args.sizes:
        for mode in args.modes:
            row = run_one(args, mode, nqubits, log_dir)
            all_rows.append(row)
            append_csv(summary_csv, [row])
            append_jsonl(summary_jsonl, [row])

            print(
                f"[{row['status']}] mode={mode} n={nqubits} "
                f"engine={row.get('expectation_engine')} "
                f"elapsed={row.get('elapsed_sec_internal')}s "
                f"rss={row.get('max_rss_kb')}KB "
                f"err={row.get('max_abs_error')} "
                f"allclose={row.get('allclose_1e-8')}",
                flush=True,
            )

    print()
    print(f"Done. Summary CSV: {summary_csv}")
    print(f"Done. Summary JSONL: {summary_jsonl}")
    print(f"Full logs are in: {log_dir}")


if __name__ == "__main__":
    main()
