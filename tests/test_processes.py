"""Real subprocess tests for launcher termination and surviving worker leases."""

import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rcuf.processes import defer_termination, exclusive_lock, termination_signals
from rcuf.runner import run_worker

LAUNCHER = r"""
import subprocess, sys
from pathlib import Path
import rcuf.launch as module
module.load_data = lambda _: ({"manifest_sha256": "fixture"}, [])
module.provenance = lambda: {"source_sha256": "fixture"}
module.require_previous_gate = lambda *args: None
run = Path(sys.argv[1])
real_popen = subprocess.Popen
def fake_worker(command, **kwargs):
    child = "from pathlib import Path; import time; from rcuf.processes import exclusive_lock; "
    child += "lease = exclusive_lock(Path(" + repr(str(run / "smoke/worker-0.lock")) + ")); "
    child += "lease.__enter__(); time.sleep(60)"
    process = real_popen([sys.executable, "-u", "-c", child], **kwargs)
    print("CHILD=" + str(process.pid), flush=True)
    return process
module.subprocess.Popen = fake_worker
module.launch(Path("."), run, "smoke", "0")
"""


def wait_for_worker(parent, run):
    child = None
    selector = selectors.DefaultSelector()
    selector.register(parent.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            for _, _ in selector.select(timeout=0.1):
                line = parent.stdout.readline()
                if not line:
                    raise AssertionError(f"Launcher exited early: {parent.poll()}")
                if line.startswith("CHILD="):
                    child = int(line.split("=")[1])
            if child is not None:
                try:
                    with exclusive_lock(run / "smoke/worker-0.lock"):
                        pass
                except RuntimeError:
                    return child
        raise AssertionError("Worker failed to acquire its lease")
    finally:
        selector.close()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_launcher_termination_reaps_children_and_releases_locks(tmp_path, signum):
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", LAUNCHER, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child = None
    try:
        child = wait_for_worker(parent, tmp_path)
        parent.send_signal(signum)
        assert parent.wait(timeout=5) == 128 + signum
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        with (
            exclusive_lock(tmp_path / "smoke/launcher.lock"),
            exclusive_lock(tmp_path / "smoke/worker-0.lock"),
        ):
            pass
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child is not None:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.stdout.close()


def test_sigkill_orphan_cannot_be_overwritten_by_restarted_worker(tmp_path):
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", LAUNCHER, str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child = None
    try:
        child = wait_for_worker(parent, tmp_path)
        parent.kill()
        parent.wait(timeout=5)
        os.kill(child, 0)  # independent-session orphan remains alive after SIGKILL
        with exclusive_lock(tmp_path / "smoke/launcher.lock"):
            pass
        with pytest.raises(RuntimeError, match="Another process already owns"):
            run_worker(Path("unused"), tmp_path, "smoke", 0, 1)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child is not None:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.stdout.close()


def test_signal_during_spawn_registration_is_deferred():
    registered = []
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as error, termination_signals(), defer_termination():
        os.kill(os.getpid(), signal.SIGTERM)
        registered.append("child registered for cleanup")
    assert registered
    assert error.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous
