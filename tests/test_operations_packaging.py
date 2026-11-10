import json
from pathlib import Path

import pytest

from chronogrid.operations import paths
from chronogrid.operations.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_asset_selection_is_explicit_and_does_not_follow_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("CHRONOGRID_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.workspace() == ROOT
    monkeypatch.setenv("CHRONOGRID_WORKSPACE", str(tmp_path))
    with pytest.raises(ValueError, match="assets unavailable"):
        paths.workspace(required=False)
    assert paths.workspace(ROOT) == ROOT


def test_installed_code_without_assets_requires_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("CHRONOGRID_WORKSPACE", raising=False)
    monkeypatch.setattr(
        paths, "__file__", str(tmp_path / "site-packages/chronogrid/operations/paths.py")
    )
    assert paths.workspace(required=False) is None
    with pytest.raises(ValueError, match="provide --workspace"):
        paths.workspace()
    assert paths.workspace(ROOT) == ROOT


def test_verify_missing_journal_never_creates_evidence(tmp_path, capsys):
    missing = tmp_path / "nested/missing.sqlite"
    with pytest.raises(SystemExit) as exc:
        main(["--db", str(missing), "verify"])
    assert exc.value.code == 2 and "does not exist" in capsys.readouterr().err
    assert not missing.parent.exists()


def test_standalone_cli_inventory_and_help_do_not_create_journal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "workspace", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    main(["doctor"])
    data = json.loads(capsys.readouterr().out)
    assert data["workspace"] is None and "readiness" in data["scope"]
    with pytest.raises(SystemExit) as exc:
        main(["verify"])
    assert exc.value.code == 2 and not (tmp_path / "data").exists()
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0 and not (tmp_path / "data").exists()


def test_worker_uses_explicit_workspace(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from chronogrid.operations import worker
    from chronogrid.operations.contracts import Run
    from chronogrid.operations.journal import Journal

    def clock():
        return datetime(2027, 1, 14, 9, tzinfo=timezone.utc)

    store = Journal(tmp_path / "journal.sqlite", clock)
    store.register(
        Run(
            run_id="packaged",
            experiment_kind="prospective",
            protocol_sha256="a" * 64,
            start_date="2027-01-15",
            end_date="2027-01-15",
            arms=("model",),
            allowed_model_sha256=("b" * 64,),
            demand_low_mw=40000,
            demand_high_mw=60000,
        )
    )
    seen = []

    def capture(*args):
        seen.append(args[-1])
        return None, "c" * 64

    monkeypatch.setattr(worker, "capture_fallback48", capture)
    result = worker.run_once(
        store, "packaged", "model", "local48", fetch=lambda _: "d" * 64, workspace=ROOT
    )
    assert result["status"] == "committed_local_only" and seen == [
        ROOT / "artifacts/experiments/local_calendar_v3"
    ]
