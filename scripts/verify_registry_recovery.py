# Repository scripts bootstrap the checkout before importing the installed package.
# ruff: noqa: E402
"""Isolated local registry migration, interruption, restore and real HTTP drill.

Uses existing ONNX bytes, never training models or modifying the historical registry.
The workload is synthetic fault/recovery evidence, not prospective load performance.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chronogrid.operations.artifact_registry import (
    ArtifactRegistry,
    RegistryConflict,
    RegistryIntegrityError,
    sha,
)


def register(registry, head):
    model = (ROOT / f"artifacts/models/{head}_model.onnx").read_bytes()
    metadata = (ROOT / f"artifacts/models/{head}_metadata.json").read_bytes()
    return registry.register(
        "registry-drill",
        model,
        metadata,
        expected_model_sha256=sha(model),
        expected_metadata_sha256=sha(metadata),
    )


def probe_http(registry_path, bundle, log):
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    token = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update(
        CHRONOGRID_ARTIFACT_REGISTRY=str(registry_path),
        CHRONOGRID_REGISTRY_MODEL="registry-drill",
        CHRONOGRID_DIAGNOSTICS_TOKEN=token,
        PYTHONDONTWRITEBYTECODE="1",
    )

    def get(path, data=None, authorized=True):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer " + token
        request = Request(
            f"http://127.0.0.1:{port}" + path,
            data=None if data is None else json.dumps(data).encode(),
            headers=headers,
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read()), response.headers

    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "chronogrid.serving.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(60):
                try:
                    get("/ready")
                    break
                except URLError:
                    if process.poll() is not None:
                        raise RuntimeError("diagnostic server exited before readiness")
                    time.sleep(0.25)
            else:
                raise RuntimeError("diagnostic server readiness timed out")
            info, _ = get("/model/info")
            assert info["champion_metadata"]["bundle_id"] == bundle
            response, headers = get(
                "/predict", dict(features=[{name: 0 for name in info["feature_names"]}])
            )
            assert (
                len(response["predictions"]) == 1
                and response["evidence_kind"] == "caller_supplied_features"
            )
            assert headers["X-ChronoGrid-Feature-Parity"] == "not-established"
            try:
                get("/model/info", authorized=False)
            except HTTPError as exc:
                assert exc.code == 401
            else:
                raise AssertionError("unauthenticated metadata unexpectedly accessible")
            return dict(
                ready=True,
                bundle_id=bundle,
                prediction_count=1,
                evidence_kind=response["evidence_kind"],
                feature_parity=headers["X-ChronoGrid-Feature-Parity"],
                unauthenticated_metadata_status=401,
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def drill(output, work):
    legacy = ROOT / "data/chronogrid.db"
    original = sha(legacy.read_bytes())
    registry = ArtifactRegistry.create(work / "registry.sqlite")
    migration = registry.import_legacy(legacy)
    assert registry.verify()["active"] == {}
    a, b = register(registry, "champion"), register(registry, "load_only")
    registry.activate("registry-drill", a, expected_current=None)
    baseline = registry.verify()
    started = time.perf_counter()
    backup = registry.backup(work / "backup.sqlite")
    backup_seconds = time.perf_counter() - started
    child = """
import os,sys
from chronogrid.operations.artifact_registry import ArtifactRegistry
r=ArtifactRegistry(sys.argv[1])
if sys.argv[4]=='before':
    r._checkpoint=lambda stage: os._exit(19) if stage=='pointer_before_event' else None
r.activate('registry-drill',sys.argv[2],expected_current=sys.argv[3])
os._exit(20)
"""
    failures = {}
    for phase, exit_code in [("before", 19), ("after", 20)]:
        result = subprocess.run(
            [sys.executable, "-c", child, str(registry.path), b, a, phase],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        (output / f"{phase}-commit-child.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        assert result.returncode == exit_code
        current = ArtifactRegistry(registry.path).verify()
        if phase == "before":
            assert current["state_sha256"] == baseline["state_sha256"]
        else:
            assert current["active"]["registry-drill"] == b
        failures[phase] = dict(process_exit_code=result.returncode, verified_state=current)
    try:
        registry.activate("registry-drill", a, expected_current=a)
    except RegistryConflict:
        stale_rejected = True
    else:
        raise AssertionError("stale activation was accepted")
    assert registry.rollback("registry-drill", expected_current=b) == a
    registry.activate("registry-drill", b, expected_current=a)
    live = registry.verify()
    started = time.perf_counter()
    restored, recovery = ArtifactRegistry.restore(
        work / "backup.sqlite", work / "restored.sqlite", expected_sha256=backup["sha256"]
    )
    restore_seconds = time.perf_counter() - started
    assert restored.verify()["state_sha256"] == baseline["state_sha256"]
    rejected = {}
    for name, target, expected in [
        ("corrupt_checksum", work / "bad.sqlite", "0" * 64),
        ("existing_target", work / "restored.sqlite", backup["sha256"]),
    ]:
        try:
            ArtifactRegistry.restore(work / "backup.sqlite", target, expected_sha256=expected)
        except (RegistryIntegrityError, FileExistsError):
            rejected[name] = True
        else:
            raise AssertionError("unsafe restore accepted")
    before = sha(restored.path.read_bytes())
    http = probe_http(restored.path, a, output / "http.log")
    assert sha(restored.path.read_bytes()) == before
    assert sha(legacy.read_bytes()) == original
    report = dict(
        status="PASS",
        created_at=datetime.now(timezone.utc).isoformat(),
        experiment_kind="synthetic_fault_and_local_recovery",
        working_directory=str(work),
        migration=migration,
        legacy_sha256_before=original,
        legacy_sha256_after=sha(legacy.read_bytes()),
        backup=backup,
        recovery=recovery,
        backup_seconds=backup_seconds,
        restore_seconds=restore_seconds,
        interruptions=failures,
        stale_activation_rejected=stale_rejected,
        unsafe_restores_rejected=rejected,
        live_state=live,
        restored_state=restored.verify(),
        http=http,
        post_backup_transitions_not_in_restored_snapshot=live["transitions"]
        - restored.verify()["transitions"],
        limitations=[
            "Local process interruption, not power-loss or hardware-failure proof",
            "No offsite/WORM backup, independent custody, production DR, or HA validation",
            "Recovery timings describe one local drill, not an SLA",
            "Diagnostic bundles do not establish legacy feature parity or calibration compatibility",
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                status="PASS",
                legacy_rows=migration["rows"],
                http=http,
                recovered_transitions=report["restored_state"]["transitions"],
                live_transitions=live["transitions"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    work = (args.workdir or ROOT / ".tools" / ("registry-drill-" + uuid4().hex)).resolve()
    work.mkdir(parents=True, exist_ok=False)
    drill(args.output.resolve(), work)
