"""One sequential all-node community-analysis run; no downloads or filtering.

Exactly one storage directory, weight mode and seed are processed. Presets are
computational budgets, not a target community count or assurance of convergence.
The orchestrator uses only Python's standard library; --help imports no data
libraries. Each scientific stage runs in the explicitly recorded interpreter.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PIPELINE_VERSION = "full-population-three-stage-1.0"
SOURCES = ("pipeline.py", "PIPELINE.md", "run_sparse.py", "run_coarse.py",
           "sparse_optimizer.py", "coarse_refine.py", "sparse_storage.py",
           "connectivity.py", "METHOD.md", "COARSE_REFINEMENT.md")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_status(path, data):
    pending = Path(str(path) + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(pending, path)


def array_stamps(storage, manifest):
    result = {}
    for name, entry in manifest["arrays"].items():
        path = (storage / entry["file"]).resolve()
        if path.parent != storage or not path.is_file():
            raise ValueError("Storage array must be an existing direct child: " + name)
        stat = path.stat()
        if stat.st_size != entry["bytes"]:
            raise ValueError("Storage array size differs from manifest: " + name)
        result[name] = {"file": path.name, "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "declared_sha256": entry["sha256"]}
    return result


def child_environment(args, source_dir):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def preflight(interpreter, env, snapshot):
    code = """import importlib, importlib.metadata, json, platform, sys
packages = [('numpy','numpy'),('numba','numba'),('duckdb','duckdb'),
            ('pyarrow','pyarrow'),('igraph','igraph'),('leidenalg','leidenalg')]
info = {'python_executable':sys.executable,'python_version':sys.version,
        'platform':platform.platform(),'packages':{}}
for distribution,name in packages:
    module=importlib.import_module(name)
    info['packages'][name]={'distribution_version':importlib.metadata.version(distribution),
                            'module_file':getattr(module,'__file__',None)}
print(json.dumps(info))
"""
    result = subprocess.run([interpreter, "-u", "-c", code], env=env,
                            cwd=snapshot, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError("Scientific environment preflight failed; nothing installed.\n" + result.stderr)
    return json.loads(result.stdout)


def run_child(command, logfile, env, snapshot):
    """Relay one subprocess; never overlap or automatically retry stages."""
    with logfile.open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=snapshot, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            raise


def check_result(stage, expected, storage_hash):
    path = Path(stage["output"]) / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["storage_manifest_sha256"] != storage_hash:
        raise ValueError("Stage population manifest changed")
    for field in ("seed", "weight_mode", "gamma", "n_wallets", "n_events", "n_links"):
        if data[field] != expected[field]:
            raise ValueError("Stage input/objective mismatch: " + field)
    if not data.get("all_nodes_assigned") or data.get("nodes_excluded") != 0 or data.get("links_excluded") != 0:
        raise ValueError("Stage did not preserve complete node/link scope")
    if not math.isfinite(data["q_barber_checked"]):
        raise ValueError("Stage quality is not finite")
    for entry in data["memberships"]:
        artifact = Path(stage["output"]) / entry["file"]
        if artifact.parent.resolve() != Path(stage["output"]).resolve() or digest(artifact) != entry["sha256"]:
            raise ValueError("Stage membership hash mismatch")
    return data, digest(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty output root; existing results are never overwritten")
    parser.add_argument("--mode", choices=("amount", "wallet_share"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--stage1-sweeps", type=int, default=30)
    parser.add_argument("--stage1-merge-rounds", type=int, default=100)
    parser.add_argument("--stage1-seconds", type=float, default=600)
    parser.add_argument("--coarse-iterations", type=int, default=30)
    parser.add_argument("--coarse-seconds", type=float, default=180)
    parser.add_argument("--final-sweeps", type=int, default=150)
    parser.add_argument("--final-merge-rounds", type=int, default=100)
    parser.add_argument("--final-seconds", type=float, default=600)
    parser.add_argument("--reserve-gib", type=float, default=3.0)
    parser.add_argument("--python", default=sys.executable, help="Interpreter used for every stage; defaults to this Python")
    parser.add_argument("--verify-storage-hashes", action="store_true", help="Also rehash all stored arrays in the first stage; adds a complete sequential read")
    args = parser.parse_args(argv)
    if not math.isfinite(args.gamma) or args.gamma < 0:
        parser.error("gamma must be finite and nonnegative")
    for field in ("stage1_sweeps", "stage1_merge_rounds", "stage1_seconds", "coarse_iterations",
                  "coarse_seconds", "final_sweeps", "final_merge_rounds", "final_seconds"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            parser.error(field.replace("_", "-") + " must be finite and nonnegative")
    if not math.isfinite(args.reserve_gib) or args.reserve_gib <= 0:
        parser.error("reserve-gib must be positive and finite")
    source = Path(__file__).resolve().parent
    storage, out = args.storage.resolve(), args.output.resolve()
    manifest_path = storage / "storage.json"
    storage_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if storage_manifest.get("status") != "validated_complete":
        raise ValueError("Only validated complete sparse storage is accepted")
    if out == storage or storage in out.parents:
        raise ValueError("Output must not be inside the input storage directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Existing output preserved; choose a new or empty directory")
    for name in SOURCES:
        if not (source / name).is_file():
            raise FileNotFoundError("Required reproducible source missing: " + name)
    env = child_environment(args, source)
    interpreter = shutil.which(args.python) or str(Path(args.python).resolve())
    if not Path(interpreter).is_file():
        raise FileNotFoundError("Requested Python interpreter does not exist")
    storage_hash = digest(manifest_path)
    stamps = array_stamps(storage, storage_manifest)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "code"
    snapshot.mkdir()
    source_hashes = {}
    for name in SOURCES:
        value = digest(source / name)
        shutil.copyfile(source / name, snapshot / name)
        if digest(snapshot / name) != value:
            raise RuntimeError("Source changed while snapshotting: " + name)
        source_hashes[name] = value
    stage1, coarse, final = (out / name for name in ("stage1", "coarse", "final"))
    common = ["--storage", str(storage), "--seed", str(args.seed), "--reserve-gib", str(args.reserve_gib)]
    sparse_common = common + ["--mode", args.mode, "--gamma", str(args.gamma)]
    first = [interpreter, "-u", str(snapshot / "run_sparse.py"), *sparse_common,
             "--output", str(stage1), "--max-sweeps", str(args.stage1_sweeps),
             "--max-merge-rounds", str(args.stage1_merge_rounds), "--max-seconds", str(args.stage1_seconds)]
    if args.verify_storage_hashes:
        first.append("--verify-storage-hashes")
    stages = [
        {"name": "stage1", "output": str(stage1), "command": first},
        {"name": "coarse", "output": str(coarse), "command": [interpreter, "-u", str(snapshot / "run_coarse.py"),
         *common, "--initial-run", str(stage1), "--output", str(coarse),
         "--max-iterations", str(args.coarse_iterations), "--max-seconds", str(args.coarse_seconds)]},
        {"name": "final", "output": str(final), "command": [interpreter, "-u", str(snapshot / "run_sparse.py"),
         *sparse_common, "--resume-from", str(coarse), "--output", str(final),
         "--max-sweeps", str(args.final_sweeps), "--max-merge-rounds", str(args.final_merge_rounds),
         "--max-seconds", str(args.final_seconds)]}]
    expected = {"seed": args.seed, "weight_mode": args.mode, "gamma": args.gamma,
                **{k: storage_manifest[k] for k in ("n_wallets", "n_events", "n_links")}}
    plan = {"pipeline_version": PIPELINE_VERSION, "created_utc": utc(),
            "storage": str(storage), "storage_manifest_sha256": storage_hash,
            "scope": storage_manifest["scope"], "input": expected,
            "storage_array_stamps": stamps, "source_sha256": source_hashes,
            "python_executable": interpreter, "pythonpath": env.get("PYTHONPATH", ""),
            "verify_all_storage_hashes_in_stage1": args.verify_storage_hashes,
            "stages": stages, "search_budgets_not_hard_total_wallclock_caps": True,
            "coarse_duckdb_memory_limit_bytes": 2 * 2 ** 30,
            "coarse_spill_limit_bytes": 4 * 2 ** 30,
            "coarse_graph_memory_budget_bytes": 3 * 2 ** 30,
            "reserve_gib": args.reserve_gib, "single_seed_only": True,
            "community_count_fixed": False, "downloads": False,
            "nodes_or_links_filtered": False, "paper_ready": False,
            "plan_note": "Budgets recorded before processing. Exactly these three sequential stages; no automatic retries, extra seeds or modes."}
    write_new_json(out / "pipeline_plan.json", plan)
    status = {"status": "preflight", "started_utc": utc(), "stages_finished": [],
              "original_optimizer_converged": False, "paper_ready": False}
    status_path = out / "pipeline_status.json"
    write_status(status_path, status)
    started = time.monotonic()
    reports = []
    try:
        software = preflight(interpreter, env, snapshot)
        write_new_json(out / "software_versions.json", software)
        for stage in stages:
            if digest(manifest_path) != storage_hash or array_stamps(storage, storage_manifest) != stamps:
                raise ValueError("Input storage changed between stages; stopped without retry")
            if any(digest(snapshot / name) != value for name, value in source_hashes.items()):
                raise ValueError("Frozen source snapshot changed between stages")
            status.update(status="running", active_stage=stage["name"])
            write_status(status_path, status)
            print("\nStarting " + stage["name"] + " for one complete population, mode=" + args.mode + ", seed=" + str(args.seed), flush=True)
            returncode = run_child(stage["command"], out / (stage["name"] + ".log"), env, snapshot)
            if returncode:
                raise RuntimeError(stage["name"] + " exited with status " + str(returncode) + "; no next stage or retry launched")
            report, result_hash = check_result(stage, expected, storage_hash)
            previous_hash = reports[-1]["result_sha256"] if reports else None
            if stage["name"] == "coarse" and report["initial_result_sha256"] != previous_hash:
                raise ValueError("Quotient stage did not use the recorded stage1 result")
            if stage["name"] == "final" and report["resume_result_sha256"] != previous_hash:
                raise ValueError("Final stage did not use the recorded quotient result")
            reports.append({"stage": stage["name"], "result": str(Path(stage["output"]) / "result.json"),
                            "result_sha256": result_hash, "q_barber_checked": report["q_barber_checked"],
                            "n_communities": report["n_communities"], "seconds": report["total_seconds"],
                            "status": report["status"], "optimizer_stop_reason": report.get("optimizer", {}).get("stop_reason"),
                            "optimizer_converged": report.get("optimizer", {}).get("converged", False)})
            status["stages_finished"] = reports.copy()
            write_status(status_path, status)
        converged = bool(report["optimizer"].get("converged", False))
        connected = report["groups_with_multiple_internal_components"] == 0
        summary = {"pipeline_version": PIPELINE_VERSION,
                   "status": "finished_original_optimizer_converged" if converged else "finished_budgets_original_optimizer_not_converged",
                   "finished_utc": utc(), "elapsed_seconds": time.monotonic() - started,
                   "storage_manifest_sha256": storage_hash, "input": expected, "stages": reports,
                   "final_result": str(final / "result.json"), "original_optimizer_converged": converged,
                   "final_optimizer_stop_reason": report["optimizer"]["stop_reason"],
                   "all_final_communities_connected": connected,
                   "single_seed_only": True, "seed_stability_tested": False,
                   "inferential_p_value": None, "paper_ready": False,
                   "scientific_status": "Preliminary descriptive partition. Convergence is not a global-optimum, stability or significance claim.",
                   "nodes_excluded": 0, "links_excluded": 0, "downloads": False}
        write_new_json(out / "pipeline_result.json", summary)
        status.update(status=summary["status"], active_stage=None,
                      original_optimizer_converged=converged, result="pipeline_result.json")
        write_status(status_path, status)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        return 0 if converged else 2
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc), finished_utc=utc(),
                      elapsed_seconds=time.monotonic() - started,
                      original_optimizer_converged=False, paper_ready=False)
        write_status(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
