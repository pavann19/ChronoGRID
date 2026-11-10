"""Real ONNX bytes, SQLite concurrency, process interruption and verified recovery."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier
from threading import Event, Thread, current_thread

import numpy as np
import pytest

from chronogrid.operations.artifact_registry import ArtifactRegistry, RegistryConflict, RegistryIntegrityError, sha
from chronogrid.serving.onnx_engine import OnnxInferenceEngine

ROOT = Path(__file__).resolve().parents[1]


def register(registry, head='champion', name='load', note=None):
    model = (ROOT / f'artifacts/models/{head}_model.onnx').read_bytes()
    metadata = (ROOT / f'artifacts/models/{head}_metadata.json').read_bytes()
    if note:
        metadata = json.dumps(json.loads(metadata) | dict(review_note=note)).encode()
    return registry.register(name, model, metadata, expected_model_sha256=sha(model), expected_metadata_sha256=sha(metadata))


@pytest.fixture
def registry(tmp_path):
    return ArtifactRegistry.create(tmp_path / 'registry.sqlite')


def test_idempotent_registration_unique_channel_and_snapshot_bytes(registry):
    a = register(registry)
    assert register(registry) == a
    registry.activate('load', a, expected_current=None)
    bundle = registry.active_bundle('load')
    pinned_engine = OnnxInferenceEngine.from_bytes(bundle['model_bytes'], bundle['metadata_bytes'])
    before = pinned_engine.predict(np.zeros((1, 43)))
    b = register(registry, 'load_only')
    registry.activate('load', b, expected_current=a)
    assert registry.active_bundle('load')['revision'] == 2
    assert len(pinned_engine.feature_names) == 43  # Already-loaded engine stays one consistent snapshot.
    np.testing.assert_array_equal(before, pinned_engine.predict(np.zeros((1, 43))))
    with registry.connection(write=True) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute('INSERT INTO active VALUES (?,?)', ('load', a))
    assert registry.rollback('load', expected_current=b) == a
    assert registry.active_bundle('load')['metadata_bytes'] == bundle['metadata_bytes']
    assert registry.verify()['transitions'] == 3


def test_checksum_schema_and_cross_model_rejection(registry):
    with pytest.raises(RegistryIntegrityError, match='checksum'):
        registry.register('load', b'bad', b'{}', expected_model_sha256='0'*64, expected_metadata_sha256='0'*64)
    a = register(registry, name='other')
    with pytest.raises(RegistryConflict, match='another'):
        registry.activate('load', a, expected_current=None)
    assert registry.verify()['active'] == {}


def test_two_competing_activations_have_one_winner(registry):
    a, b, c = register(registry), register(registry, note='B'), register(registry, note='C')
    registry.activate('load', a, expected_current=None)
    barrier = Barrier(2)
    def compete(bundle):
        barrier.wait()
        try:
            ArtifactRegistry(registry.path).activate('load', bundle, expected_current=a)
            return 'activated'
        except RegistryConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compete, (b, c)))
    assert sorted(results) == ['activated', 'conflict']
    assert registry.verify()['transitions'] == 2


@pytest.mark.parametrize('stage', ['pointer_before_event', 'transition_before_commit'])
def test_process_death_rolls_back_uncommitted_pointer_and_history(registry, stage):
    a, b = register(registry), register(registry, note='B')
    registry.activate('load', a, expected_current=None)
    before = registry.verify()['state_sha256']
    code = """
import os,sys
from chronogrid.operations.artifact_registry import ArtifactRegistry
r=ArtifactRegistry(sys.argv[1])
r._checkpoint=lambda stage: os._exit(19) if stage==sys.argv[4] else None
r.activate('load',sys.argv[2],expected_current=sys.argv[3])
"""
    result = subprocess.run([sys.executable, '-c', code, str(registry.path), b, a, stage], cwd=ROOT, timeout=30)
    assert result.returncode == 19
    assert ArtifactRegistry(registry.path).verify()['state_sha256'] == before


def test_process_exit_after_commit_preserves_new_identity(registry):
    a, b = register(registry), register(registry, note='B')
    registry.activate('load', a, expected_current=None)
    code = "import os,sys; from chronogrid.operations.artifact_registry import ArtifactRegistry; ArtifactRegistry(sys.argv[1]).activate('load',sys.argv[2],expected_current=sys.argv[3]); os._exit(20)"
    result = subprocess.run([sys.executable, '-c', code, str(registry.path), b, a], cwd=ROOT, timeout=30)
    assert result.returncode == 20
    assert ArtifactRegistry(registry.path).active_bundle('load')['bundle_id'] == b


def test_backup_restore_is_exact_logical_snapshot_and_never_overwrites(registry, tmp_path):
    a = register(registry)
    registry.activate('load', a, expected_current=None)
    snapshot = registry.backup(tmp_path / 'backup.sqlite')
    b = register(registry, 'load_only')
    registry.activate('load', b, expected_current=a)
    restored, report = ArtifactRegistry.restore(tmp_path/'backup.sqlite', tmp_path/'restored.sqlite',
                                               expected_sha256=snapshot['sha256'])
    assert report['verified_state']['state_sha256'] == snapshot['verified_state']['state_sha256']
    assert restored.active_bundle('load')['bundle_id'] == a
    assert registry.active_bundle('load')['bundle_id'] == b  # No live-pointer rollback on restore.
    with pytest.raises(FileExistsError):
        registry.backup(tmp_path/'backup.sqlite')
    with pytest.raises(FileExistsError):
        ArtifactRegistry.restore(tmp_path/'backup.sqlite', tmp_path/'restored.sqlite', expected_sha256=snapshot['sha256'])
    with pytest.raises(RegistryIntegrityError, match='checksum'):
        ArtifactRegistry.restore(tmp_path/'backup.sqlite', tmp_path/'bad.sqlite', expected_sha256='0'*64)
    assert not (tmp_path/'bad.sqlite').exists()


def test_pointer_tampering_fails_read_and_backup(registry, tmp_path):
    a, b = register(registry), register(registry, note='B')
    registry.activate('load', a, expected_current=None)
    with registry.connection(write=True) as db:
        db.execute('UPDATE active SET bundle_id=?', (b,))
    with pytest.raises(RegistryIntegrityError, match='pointer'):
        registry.active_bundle('load')
    with pytest.raises(RegistryIntegrityError):
        registry.backup(tmp_path/'untrusted.sqlite')
    assert not (tmp_path/'untrusted.sqlite').exists()


def test_blob_updates_and_deletes_are_rejected_by_database(registry):
    register(registry)
    for sql in ('UPDATE blobs SET content=?', 'DELETE FROM blobs WHERE sha256=?'):
        with registry.connection(write=True) as db:
            value = b'bad' if sql.startswith('UPDATE') else db.execute('SELECT sha256 FROM blobs LIMIT 1').fetchone()[0]
            with pytest.raises(sqlite3.IntegrityError, match='immutable'):
                db.execute(sql, (value,))


def test_failed_registration_leaves_no_half_bundle_or_blobs(registry, monkeypatch):
    def fail(stage):
        if stage == 'registered_before_commit':
            raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(registry,'_checkpoint',fail)
    with pytest.raises(RuntimeError):
        register(registry)
    assert registry.verify()['bundles'] == 0
    with registry.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM blobs').fetchone()[0] == 0


def test_migration_quarantines_duplicate_legacy_champions_without_touching_source(registry, tmp_path):
    source=tmp_path/'legacy.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE model_registry(model_id TEXT PRIMARY KEY,is_champion INTEGER)')
        db.executemany('INSERT INTO model_registry VALUES (?,1)', [('one',),('two',)])
    before=sha(source.read_bytes())
    assert registry.import_legacy(source)['activated'] == 0
    registry.import_legacy(source)
    assert registry.verify()['legacy_records'] == 2
    assert registry.verify()['active'] == {}
    assert sha(source.read_bytes()) == before


def test_malformed_legacy_identity_is_not_silently_dropped(registry,tmp_path):
    source=tmp_path/'bad-legacy.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE model_registry(model_id TEXT)')
        db.execute('INSERT INTO model_registry VALUES (NULL)')
    with pytest.raises(sqlite3.IntegrityError):
        registry.import_legacy(source)
    assert registry.verify()['legacy_records']==0


def test_backup_keeps_one_snapshot_while_writer_commits(registry, tmp_path, monkeypatch):
    a,b=register(registry),register(registry,note='B')
    registry.activate('load',a,expected_current=None)
    expected=registry.verify()['state_sha256']
    ready,changed=Event(),Event()
    original=ArtifactRegistry._verify
    def inspect(self,db):
        result=original(self,db)
        if current_thread().name=='snapshot-backup' and self.path==registry.path:
            ready.set()
            assert changed.wait(10)
        return result
    monkeypatch.setattr(ArtifactRegistry,'_verify',inspect)
    results=[]
    errors=[]
    def copy():
        try:results.append(registry.backup(tmp_path/'concurrent.sqlite'))
        except BaseException as exc:errors.append(exc)
    thread=Thread(target=copy,name='snapshot-backup')
    thread.start()
    try:
        assert ready.wait(10)
        registry.activate('load',b,expected_current=a)
    finally:
        changed.set()
        thread.join(15)
    assert not thread.is_alive() and not errors
    assert results[0]['verified_state']['state_sha256']==expected
    assert registry.active_bundle('load')['bundle_id']==b


def test_backup_target_race_preserves_other_callers_file(registry,tmp_path,monkeypatch):
    import os
    a=register(registry)
    registry.activate('load',a,expected_current=None)
    original=os.link
    target=tmp_path/'race.sqlite'
    def race(source,destination):
        target.write_bytes(b'other caller won')
        original(source,destination)
    monkeypatch.setattr(os,'link',race)
    with pytest.raises(FileExistsError):
        registry.backup(target)
    assert target.read_bytes()==b'other caller won'
    assert not list(tmp_path.glob('.registry-*'))
