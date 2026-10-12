"""POSIX process leases and orderly handling of launcher termination."""

from __future__ import annotations

import fcntl
import signal
from contextlib import contextmanager
from pathlib import Path

TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_defer_depth = 0
_pending_signal: int | None = None


def _exit_for_signal(signum):
    for sig in TERMINATION_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    raise SystemExit(128 + signum)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process already owns {path}.") from exc
        yield


@contextmanager
def termination_signals():
    previous = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}

    def terminate(signum, frame):
        global _pending_signal
        # Let the launcher's finally block finish even if the user sends a second signal.
        if _defer_depth:
            _pending_signal = signum
            return
        _exit_for_signal(signum)

    try:
        for sig in TERMINATION_SIGNALS:
            signal.signal(sig, terminate)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def defer_termination():
    global _defer_depth, _pending_signal
    # Avoid receiving SIGTERM after Popen creates a child but before it is registered
    # in the list that the launcher will clean up.
    # Defer Python's exit instead of blocking OS signals: a child must not inherit
    # a blocked SIGTERM mask across exec.
    _defer_depth += 1
    try:
        yield
    finally:
        _defer_depth -= 1
        if not _defer_depth and _pending_signal is not None:
            signum, _pending_signal = _pending_signal, None
            _exit_for_signal(signum)
