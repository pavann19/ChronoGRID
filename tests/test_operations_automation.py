from datetime import date, datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

from chronogrid.operations import capture
from chronogrid.operations.automation import bounded_cycle, host_lock
from chronogrid.operations.contracts import Run
from chronogrid.operations.journal import Journal, digest


def registered(tmp_path, *, kind='prospective', pinned=None):
    store = Journal(tmp_path / 'journal.sqlite', clock=lambda: datetime(2027, 1, 1, tzinfo=timezone.utc))
    store.register(Run(run_id='automation', experiment_kind=kind, protocol_sha256='a'*64,
                       start_date=date(2027, 1, 15), end_date=date(2027, 1, 15),
                       arms=('baseline',), allowed_model_sha256=(pinned or digest(Path(capture.__file__).read_bytes()),),
                       demand_low_mw=90, demand_high_mw=110))
    return store


def call(store, tmp_path, runner, **kwargs):
    return bounded_cycle(store.path, 'automation', {'baseline': 'weekly'},
                         tmp_path / 'attempts', 'cycle-1', runner=runner, **kwargs)


def child_result(status='already_committed', returncode=0):
    # Actual offline child process; synthetic worker output, not forecast evidence.
    def execute(command, **kwargs):
        payload = json.dumps({'forecasts': {'baseline': {'status': status}}})
        code = f'import sys;print({payload!r});sys.exit({returncode})'
        return subprocess.run([sys.executable, '-c', code], **kwargs)
    return execute


def test_attempt_is_recorded_once_and_database_unchanged(tmp_path):
    store = registered(tmp_path)
    before = digest(store.path.read_bytes())
    result = call(store, tmp_path, child_result())
    assert result['status'] == 'completed' and not result['attention_required']
    assert json.loads(Path(result['result']).read_text()) == result
    assert json.loads(Path(result['started']).read_text())['timeout_seconds'] == 300
    assert call(store, tmp_path, lambda *a, **kw: pytest.fail('duplicate execution'))['status'] == 'already_attempted'
    assert digest(store.path.read_bytes()) == before


@pytest.mark.parametrize('status,code,expected', [('missed_gate', 0, 'needs_attention'),
                                                ('failed', 0, 'needs_attention'),
                                                ('failed', 7, 'failed')])
def test_child_failures_are_visible(tmp_path, status, code, expected):
    result = call(registered(tmp_path), tmp_path, child_result(status, code))
    assert result['status'] == expected and result['attention_required']


def test_real_child_timeout_preserves_attempt_and_releases_lock(tmp_path):
    store = registered(tmp_path)
    def slow(command, **kwargs):
        return subprocess.run([sys.executable, '-c', 'import time;time.sleep(30)'], **kwargs)
    result = call(store, tmp_path, slow, timeout_seconds=1)
    assert result['status'] == 'timed_out' and result['attention_required']
    assert call(store, tmp_path, child_result())['status'] == 'already_attempted'
    # A different explicit attempt can acquire the lock; no lease expiry needed.
    retry = bounded_cycle(store.path, 'automation', {'baseline': 'weekly'},
                          tmp_path / 'attempts', 'cycle-2', runner=child_result())
    assert retry['status'] == 'completed'


def test_incomplete_attempt_is_not_replayed(tmp_path):
    store = registered(tmp_path)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        call(store, tmp_path, interrupted)
    result = call(store, tmp_path, child_result())
    assert result['status'] == 'already_attempted' and result['result'] is None


@pytest.mark.parametrize('kind,pinned', [('synthetic_fault', None), ('prospective', 'b'*64)])
def test_preflight_rejects_replay_and_changed_weekly_identity(tmp_path, kind, pinned):
    store = registered(tmp_path, kind=kind, pinned=pinned)
    with pytest.raises(ValueError):
        call(store, tmp_path, lambda *a, **kw: pytest.fail('must not run'))
    assert not (tmp_path / 'attempts').exists()


def test_missing_journal_is_not_initialized(tmp_path):
    with pytest.raises(FileNotFoundError):
        bounded_cycle(tmp_path / 'missing.sqlite', 'run', {}, tmp_path / 'attempts', 'key')
    assert not (tmp_path / 'missing.sqlite').exists()


def test_cross_process_overlap_rejected_and_lock_released_after_exit(tmp_path):
    lock = tmp_path / 'worker.lock'
    code = ('import sys;from pathlib import Path;from chronogrid.operations.automation import host_lock;'
            '\nwith host_lock(Path(sys.argv[1])): pass')
    with host_lock(lock):
        result = subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True)
        assert result.returncode != 0
    assert subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True).returncode == 0
    killed = code.replace('pass', '__import__("os")._exit(19)')
    assert subprocess.run([sys.executable, '-c', killed, str(lock)], capture_output=True).returncode == 19
    assert subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True).returncode == 0


def test_invalid_output_is_preserved_and_flagged(tmp_path):
    def malformed(command, **kwargs):
        return subprocess.run([sys.executable, '-c', 'print("invalid JSON")'], **kwargs)
    result = call(registered(tmp_path), tmp_path, malformed)
    assert result['status'] == 'failed' and result['attention_required']
    assert Path(result['started']).with_suffix('').with_suffix('.stdout.log').read_text().strip() == 'invalid JSON'


def test_real_worker_cycle_commits_once_with_offline_source(tmp_path):
    from datetime import datetime, timedelta, timezone
    from chronogrid.operations.journal import encode
    from chronogrid.operations.worker import run_cycle
    store = registered(tmp_path)
    issued = datetime(2027, 1, 14, 10, tzinfo=timezone.utc)
    store.clock = lambda: issued
    # The fixture was registered before simulated issuance; these are synthetic test inputs.
    rows = [dict(timestamp=(issued-timedelta(hours=i+2)).isoformat(), load_mw=100+i)
            for i in reversed(range(336))]
    snapshot = store.snapshot(encode(dict(format='smard-hourly-mw-v1', unit='MW', rows=rows)),
                              'smard-normalized-hourly', issued)
    def worker(command, **kwargs):
        payload = run_cycle(store, 'automation', {'baseline': 'weekly'}, fetch=lambda _: snapshot)
        kwargs['stdout'].write(json.dumps(payload).encode())
        return subprocess.CompletedProcess(command, 0)
    result = call(store, tmp_path, worker)
    assert result['forecasts']['baseline']['status'] == 'committed_local_only'
    assert len(store.rows('forecast')) == 1
    next_result = bounded_cycle(store.path, 'automation', {'baseline': 'weekly'},
                                tmp_path / 'attempts', 'cycle-2', runner=worker)
    assert next_result['forecasts']['baseline']['status'] == 'already_committed'
    assert len(store.rows('forecast')) == 1


def test_cli_models_file_propagates_attention_exit(tmp_path, monkeypatch, capsys):
    from chronogrid.operations.cli import main
    store = registered(tmp_path)
    config = tmp_path / 'models.json'
    config.write_text('{"baseline":"weekly"}', encoding='utf-8-sig')
    def stub(db, run_id, models, evidence_dir, cycle_key, **kwargs):
        assert models == {'baseline': 'weekly'}
        return {'status': 'needs_attention', 'attention_required': True}
    monkeypatch.setattr('chronogrid.operations.automation.bounded_cycle', stub)
    with pytest.raises(SystemExit) as exc:
        main(['--db', str(store.path), 'bounded-cycle', 'automation',
              '--models-file', str(config), '--evidence-dir', str(tmp_path / 'attempts'),
              '--cycle-key', 'test'])
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)['attention_required']


def test_powershell_wrapper_preserves_paths_and_failure_exit(tmp_path):
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if shell is None:
        pytest.skip('PowerShell wrapper requires PowerShell')
    fixture = tmp_path / 'fixture with spaces'
    fixture.mkdir()
    store = registered(fixture, pinned='b'*64)
    config = fixture / 'models with spaces.json'
    config.write_text('{"baseline":"weekly"}', encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-File',
                             str(root / 'scripts/invoke_bounded_cycle.ps1'),
                             '-Python', sys.executable, '-Workspace', str(root),
                             '-Journal', str(store.path), '-RunId', 'automation',
                             '-ModelsFile', str(config), '-EvidenceDirectory', str(fixture / 'attempts')],
                            capture_output=True, timeout=30)
    assert result.returncode != 0
    assert b'weekly implementation differs' in result.stderr
    assert not (fixture / 'attempts').exists()
