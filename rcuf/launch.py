"""One independent CUDA worker per device, with visible errors and durable logs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .data import load_data
from .io import bind_metadata, provenance
from .processes import defer_termination, exclusive_lock, termination_signals
from .runner import load_selection, require_previous_gate


def launch(data_dir: Path, run_dir: Path, stage: str, devices: str, *, confirmation=False) -> None:
    selected_devices = [device.strip() for device in devices.split(",")]
    if any(not device for device in selected_devices) or len(set(selected_devices)) != len(selected_devices):
        raise ValueError("Specify distinct GPU IDs, e.g. --devices 0,1.")
    if confirmation:
        from .confirm_data import load_confirmation

        if len(selected_devices) != 2:
            raise ValueError("Confirmation needs exactly two distinct physical GPUs.")
        manifest, _ = load_confirmation(data_dir)
    else:
        manifest, _ = load_data(data_dir)
        require_previous_gate(run_dir, stage, manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    bind_metadata(
        run_dir / "experiment.json",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": provenance()["source_sha256"],
        },
    )
    if confirmation:
        # Device order stays fixed: crossover-b reverses the document assignment.
        bind_metadata(run_dir / "confirmation-devices.json", {"ordered_devices": selected_devices})
    elif stage in {"compare", "delay"}:
        load_selection(run_dir, manifest)
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(stage_dir / "launcher.lock"), termination_signals():
        processes = []
        streams = []
        try:
            for shard, device in enumerate(selected_devices):
                stream = (stage_dir / f"worker-{shard}.log").open("a", encoding="utf-8")
                streams.append(stream)
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=device.strip(),
                    PYTHONUNBUFFERED="1",
                    TOKENIZERS_PARALLELISM="false",
                )
                command = [
                    sys.executable,
                    "-m",
                    "rcuf.cli",
                    "confirm-worker" if confirmation else "worker",
                    "--data-dir",
                    str(data_dir),
                    "--run-dir",
                    str(run_dir),
                    "--phase" if confirmation else "--stage",
                    stage,
                    "--shard",
                    str(shard),
                ]
                if not confirmation:
                    command.extend(["--shards", str(len(selected_devices))])
                with defer_termination():
                    process = subprocess.Popen(
                        command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    processes.append(process)
                print(f"{stage}: GPU {device} -> shard {shard}; log {stream.name}", flush=True)
            last_status = 0.0
            while any(process.poll() is None for process in processes):
                failed = [process for process in processes if process.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(
                        "A worker failed. See its worker log; other workers are being stopped."
                    )
                now = time.monotonic()
                if now - last_status >= 30:
                    statuses = [
                        "running" if process.poll() is None else f"exit={process.returncode}"
                        for process in processes
                    ]
                    print(f"{stage} worker status: {statuses}", flush=True)
                    last_status = now
                time.sleep(1)
            if any(process.returncode for process in processes):
                raise RuntimeError("A worker failed. Read worker-*.log.")
            summarize = "confirm-summarize" if confirmation else f"summarize --stage {stage}"
            print(
                f"{stage} complete. Run rcuf {summarize} --data-dir {data_dir} --run-dir {run_dir}",
                flush=True,
            )
        finally:
            for process in processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            for stream in streams:
                stream.close()
