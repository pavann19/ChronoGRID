"""Single-host, bounded worker invocation with exclusive attempt evidence."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from chronogrid.operations.journal import digest
from chronogrid.operations.status import ReadOnlyJournal, journal_status


@contextmanager
def host_lock(path):
    """Process-lifetime lock; never infer a dead writer from wall-clock age."""
    with Path(path).open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def bounded_cycle(
    db,
    run_id,
    models,
    evidence_dir,
    cycle_key,
    *,
    workspace=None,
    timeout_seconds=300,
    runner=subprocess.run,
):
    """One cycle, no automatic retry/backfill, no scheduler or notification installation.

    A started attempt is never automatically replayed, even if the parent crashed.
    The journal remains the authority for which predictions actually committed.
    """
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or not 1 <= timeout_seconds <= 1800
    ):
        raise ValueError("timeout must be an integer from 1 to 1800 seconds")
    if not isinstance(cycle_key, str) or not cycle_key.strip() or len(cycle_key) > 128:
        raise ValueError("explicit nonempty cycle key required (maximum 128 characters)")
    db = Path(db).resolve()
    # Read-only preflight: never initialize a missing database or mutate a replay.
    journal_status(db, run_id=run_id)
    run = ReadOnlyJournal(db).get("run:" + run_id)
    if run["experiment_kind"] != "prospective":
        raise ValueError("automation requires a prospective registered run")
    if (
        not isinstance(models, dict)
        or set(models) != set(run["arms"])
        or any(m not in ("weekly", "fallback48", "adaptive48", "local48") for m in models.values())
    ):
        raise ValueError("explicit supported model required for each registered arm")
    pinned = {a["arm"]: a["model_sha256"] for a in run.get("arm_models", [])}
    if set(pinned) != set(models):
        raise ValueError("registered per-arm model pins required")
    from chronogrid.operations import capture

    for arm, model in models.items():
        if model == "weekly" and pinned[arm] != digest(Path(capture.__file__).read_bytes()):
            raise ValueError("weekly implementation differs from registered model; use a new run")
    if any(m != "weekly" for m in models.values()):
        from chronogrid.operations.paths import workspace as resolve

        workspace = resolve(workspace)
    elif workspace is not None:
        workspace = Path(workspace).resolve()
    root = Path(evidence_dir).resolve()
    # Names are digests, never user-provided path components.
    namespace = hashlib.sha256(os.path.normcase(str(db)).encode()).hexdigest()
    root = root / namespace
    root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps([run_id, cycle_key]).encode()).hexdigest()
    prefix = root / identity
    started = prefix.with_suffix(".started.json")
    result_path = prefix.with_suffix(".result.json")
    with host_lock(root / "worker.lock"):
        if started.exists():
            return dict(
                status="already_attempted",
                attention_required=True,
                started=str(started),
                result=str(result_path) if result_path.exists() else None,
                reason="Inspect existing result and journal; no automatic replay",
            )
        command = [sys.executable, "-m", "chronogrid.operations.cli", "--db", str(db)]
        if workspace is not None:
            command += ["--workspace", str(workspace)]
        command += ["run-cycle", run_id, "--models", json.dumps(models, sort_keys=True)]
        write_new(
            started,
            dict(
                format="chronogrid-bounded-cycle-v1",
                run_id=run_id,
                cycle_key=cycle_key,
                db=str(db),
                models=models,
                started_at=datetime.now(timezone.utc).isoformat(),
                timeout_seconds=timeout_seconds,
                command=command,
            ),
        )
        result = dict(
            status="failed",
            attention_required=True,
            run_id=run_id,
            cycle_key=cycle_key,
            started=str(started),
            result=str(result_path),
            production_approved=False,
        )
        with (
            prefix.with_suffix(".stdout.log").open("xb") as stdout,
            prefix.with_suffix(".stderr.log").open("xb") as stderr,
        ):
            try:
                completed = runner(
                    command, stdout=stdout, stderr=stderr, timeout=timeout_seconds, check=False
                )
                result["returncode"] = completed.returncode
                if completed.returncode == 0:
                    stdout.flush()
                    raw = prefix.with_suffix(".stdout.log").read_bytes()
                    if len(raw) > 1_000_000:
                        raise ValueError("worker response exceeds JSON inspection budget")
                    payload = json.loads(raw)
                    forecasts = payload["forecasts"]
                    if set(forecasts) != set(models):
                        raise ValueError("worker did not report every registered arm")
                    healthy = {
                        "committed_local_only",
                        "already_committed",
                        "concurrent_origin_already_committed",
                        "outside_registered_window",
                    }
                    okay = all(v["status"] in healthy for v in forecasts.values())
                    result.update(
                        status="completed" if okay else "needs_attention",
                        attention_required=not okay,
                        forecasts=forecasts,
                    )
            except subprocess.TimeoutExpired:
                result.update(
                    status="timed_out",
                    reason="Child terminated; inspect journal for partial commits",
                )
            except Exception as exc:
                result.update(status="failed", reason=str(exc))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_new(result_path, result)
        return result
