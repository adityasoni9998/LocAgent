import argparse
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datasets import load_dataset
import pandas as pd


APPTAINER_CACHEDIR = os.environ.get("APPTAINER_CACHEDIR", "/data/user_data/adityabs/apptainer_cache")
LOCALIZATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCALIZATION_DIR.parent
HOST_PATCH = Path("/usr/bin/patch")
HOST_PATCH_LIBS = [
    Path("/lib64/libattr.so.1"),
    Path("/lib64/libselinux.so.1"),
    Path("/lib64/libc.so.6"),
    Path("/lib64/libpcre2-8.so.0"),
    Path("/lib64/ld-linux-x86-64.so.2"),
]

def stage_host_patch_tool():
    if not HOST_PATCH.exists():
        raise FileNotFoundError(f"Host patch binary not found: {HOST_PATCH}")
    for lib in HOST_PATCH_LIBS:
        if not lib.exists():
            raise FileNotFoundError(f"Host patch dependency not found: {lib}")


def resolve_sif_path(instance):
    image_name = instance["image_name"].split("/")[-1]
    sif_path = Path(APPTAINER_CACHEDIR) / f"43376f1-93c33d0-{image_name}-source-minimal.sif"
    if not sif_path.exists():
        raise FileNotFoundError(f"Apptainer image not found: {sif_path}")
    return sif_path


def localization_processing(instance):
    instance_id = instance["instance_id"]
    try:
        sif_path = resolve_sif_path(instance)
    except FileNotFoundError:
        return instance_id, {}

    ground_truth_patch = instance["patch"]
    random_id = str(uuid.uuid4())
    patch_file_path = f"/tmp/{random_id}.patch"
    with open(patch_file_path, "w", newline="") as f:
        f.write(ground_truth_patch)

    overlay_root = Path(f"/tmp/overlay_{random_id}")
    (overlay_root / "upper").mkdir(parents=True, exist_ok=True)
    (overlay_root / "work").mkdir(parents=True, exist_ok=True)

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python_executable = venv_python if venv_python.exists() else Path(sys.executable)
    command = (
        # "git -C /testbed reset --hard || true; "
        # "git -C /testbed clean -fd || true; "
        "/host_tools/lib64/ld-linux-x86-64.so.2 --library-path /host_tools/lib64 "
        "/host_tools/patch --version >/tmp/patch_version.txt || true; "
        f"{python_executable} {LOCALIZATION_DIR / 'localization_patch_processing.py'} --id {random_id}"
    )
    apptainer_cmd = [
        "apptainer",
        "exec",
        "--fakeroot",
        "--cleanenv",
        "--overlay",
        str(overlay_root),          # replaces "--writable-tmpfs"
        "--bind",
        f"{REPO_ROOT}:{REPO_ROOT}",
        "--bind",
        "/tmp:/tmp",
        "--bind",
        f"{HOST_PATCH}:/host_tools/patch:ro",
        "--bind",
        "/lib64/libattr.so.1:/host_tools/lib64/libattr.so.1:ro",
        "--bind",
        "/lib64/libselinux.so.1:/host_tools/lib64/libselinux.so.1:ro",
        "--bind",
        "/lib64/libc.so.6:/host_tools/lib64/libc.so.6:ro",
        "--bind",
        "/lib64/libpcre2-8.so.0:/host_tools/lib64/libpcre2-8.so.0:ro",
        "--bind",
        "/lib64/ld-linux-x86-64.so.2:/host_tools/lib64/ld-linux-x86-64.so.2:ro",
        str(sif_path),
        "env",
        f"PYTHONPATH={LOCALIZATION_DIR}",
        "bash",
        "-c",
        command,
    ]

    result = subprocess.run(apptainer_cmd, text=True, capture_output=True, check=False)

    # Try processing the output from the JSON log
    try:
        with open(f"/tmp/{random_id}_edited_locations.json", "r") as f:
            edited_locations = json.load(f)
    except Exception:
        edited_locations = {}
    edited_locations["apptainer_stdout"] = result.stdout
    # edited_locations["apptainer_stderr"] = result.stderr
    cleanup_tmp_files(random_id)
    cleanup_overlay(overlay_root)
    return instance_id, edited_locations

def cleanup_overlay(overlay_root):
    import shutil
    shutil.rmtree(overlay_root, ignore_errors=True)

def cleanup_tmp_files(random_id):
    for suffix in (".patch", "_edited_locations.json"):
        try:
            Path(f"/tmp/{random_id}{suffix}").unlink()
        except FileNotFoundError:
            pass

def load_processed_instance_ids(output_path):
    processed = set()
    if not output_path.exists():
        return processed
    with output_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = row.get("instance_id")
            if instance_id:
                processed.add(instance_id)
    return processed


def write_jsonl_row(handle, instance_id, edited_locations):
    handle.write(json.dumps({
        "instance_id": instance_id,
        "edited_locations": edited_locations,
    }) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--output", default=str(LOCALIZATION_DIR / "edited_locations.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    stage_host_patch_tool()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("nebius/SWE-rebench", split="filtered")
    instances = [instance for instance in dataset]
    if args.limit:
        instances = instances[: args.limit]

    processed = load_processed_instance_ids(output_path)
    instances = [instance for instance in instances if instance["instance_id"] not in processed]
    print(
        f"total={len(instances)} already_done={len(processed)} remaining={len(instances)} "
        f"workers={args.workers} output={output_path}",
        file=sys.stderr,
        flush=True,
    )

    completed = 0
    successes = 0
    with output_path.open("a") as output_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(localization_processing, instance) for instance in instances]
            for future in as_completed(futures):
                instance_id, edited_locations = future.result()
                write_jsonl_row(output_handle, instance_id, edited_locations)
                completed += 1
                if edited_locations:
                    successes += 1
                if completed % 100 == 0 or completed == len(futures):
                    print(
                        f"completed={completed}/{len(futures)} non_empty={successes}",
                        file=sys.stderr,
                        flush=True,
                    )

if __name__ == "__main__":
    main()
