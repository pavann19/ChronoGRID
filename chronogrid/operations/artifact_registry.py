"""Local diagnostic artifact registry: one SQLite snapshot binds bytes and active pointer.

This is not a production promotion authority. Owner-writable SQLite and backups
remain within the local trust boundary; no independent durability is claimed.
"""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

APP_ID = 0x43474152
SCHEMA = """
CREATE TABLE blobs(sha256 TEXT PRIMARY KEY, content BLOB NOT NULL);
CREATE TABLE bundles(bundle_id TEXT PRIMARY KEY, model_name TEXT NOT NULL,
 revision INTEGER NOT NULL CHECK(revision>0), manifest TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(model_name,revision), UNIQUE(model_name,bundle_id));
CREATE TABLE active(model_name TEXT PRIMARY KEY, bundle_id TEXT NOT NULL,
 FOREIGN KEY(model_name,bundle_id) REFERENCES bundles(model_name,bundle_id));
CREATE TABLE transitions(seq INTEGER PRIMARY KEY, model_name TEXT NOT NULL,
 previous_bundle TEXT, bundle_id TEXT NOT NULL, action TEXT NOT NULL,
 recorded_at TEXT NOT NULL, previous_hash TEXT NOT NULL, sha256 TEXT NOT NULL,
 FOREIGN KEY(model_name,bundle_id) REFERENCES bundles(model_name,bundle_id));
CREATE TABLE legacy_records(source_sha256 TEXT NOT NULL, model_id TEXT NOT NULL,
 payload TEXT NOT NULL, PRIMARY KEY(source_sha256,model_id));
"""


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class RegistryConflict(ValueError):
    pass


class RegistryIntegrityError(ValueError):
    pass


class ArtifactRegistry:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError("registry does not exist; explicit creation required")
        with self.connection() as db:
            if (
                db.execute("PRAGMA application_id").fetchone()[0] != APP_ID
                or db.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise RegistryIntegrityError(
                    "unsupported registry schema; never open legacy DB as a new registry"
                )

    @classmethod
    def create(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb"):
            pass
        db = sqlite3.connect(path)
        try:
            db.executescript("BEGIN IMMEDIATE;\n" + SCHEMA)
            for table in ("blobs", "bundles", "transitions", "legacy_records"):
                for action in ("UPDATE", "DELETE"):
                    db.execute(
                        f"CREATE TRIGGER immutable_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'immutable registry evidence'); END"
                    )
            db.execute(f"PRAGMA application_id={APP_ID}")
            db.execute("PRAGMA user_version=1")
            db.commit()
            db.execute("PRAGMA journal_mode=WAL")
        finally:
            db.close()
        return cls(path)

    @contextmanager
    def connection(self, write=False):
        uri = (
            "file:" + quote(self.path.as_posix(), safe="/:") + ("?mode=rw" if write else "?mode=ro")
        )
        db = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _checkpoint(self, stage):
        """No-op seam for isolated transaction-failure tests."""

    def register(
        self,
        model_name,
        model_bytes,
        metadata_bytes,
        *,
        expected_model_sha256,
        expected_metadata_sha256,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", model_name):
            raise ValueError("invalid model name")
        if (
            sha(model_bytes) != expected_model_sha256
            or sha(metadata_bytes) != expected_metadata_sha256
        ):
            raise RegistryIntegrityError("artifact checksum mismatch before registration")
        # Validate the immutable bytes, never a path reopened after checksum checking.
        from chronogrid.serving.onnx_engine import OnnxInferenceEngine

        OnnxInferenceEngine.from_bytes(model_bytes, metadata_bytes)
        manifest = dict(
            format="chronogrid-diagnostic-bundle-v1",
            model_name=model_name,
            model_sha256=sha(model_bytes),
            metadata_sha256=sha(metadata_bytes),
            purpose="caller-feature diagnostics; not prospective or production approval",
        )
        bundle_id = sha(encode(manifest))
        with self.connection(write=True) as db:
            self._verify(db)
            existing = db.execute(
                "SELECT bundle_id FROM bundles WHERE bundle_id=?", (bundle_id,)
            ).fetchone()
            if existing:
                return bundle_id
            for raw in (model_bytes, metadata_bytes):
                db.execute(
                    "INSERT INTO blobs VALUES (?,?) ON CONFLICT(sha256) DO NOTHING", (sha(raw), raw)
                )
            revision = db.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM bundles WHERE model_name=?", (model_name,)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO bundles VALUES (?,?,?,?,?)",
                (
                    bundle_id,
                    model_name,
                    revision,
                    encode(manifest).decode(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._checkpoint("registered_before_commit")
        return bundle_id

    def _bundle(self, db, bundle_id):
        row = db.execute("SELECT * FROM bundles WHERE bundle_id=?", (bundle_id,)).fetchone()
        if row is None:
            raise KeyError("unknown artifact bundle")
        manifest = json.loads(row["manifest"])
        if sha(encode(manifest)) != bundle_id or manifest["model_name"] != row["model_name"]:
            raise RegistryIntegrityError("bundle manifest identity mismatch")
        result = dict(row)
        for role in ("model", "metadata"):
            expected = manifest[role + "_sha256"]
            blob = db.execute("SELECT content FROM blobs WHERE sha256=?", (expected,)).fetchone()
            if blob is None or sha(blob["content"]) != expected:
                raise RegistryIntegrityError("bundle content checksum mismatch")
            result[role + "_bytes"] = blob["content"]
        result["manifest"] = manifest
        return result

    def activate(self, model_name, bundle_id, *, expected_current):
        return self._transition(model_name, bundle_id, expected_current, "activate")

    def rollback(self, model_name, *, expected_current):
        return self._transition(model_name, None, expected_current, "rollback")

    def _transition(self, name, bundle_id, expected, action):
        with self.connection(write=True) as db:
            self._verify(db)
            current = db.execute(
                "SELECT bundle_id FROM active WHERE model_name=?", (name,)
            ).fetchone()
            current = current[0] if current else None
            if current != expected:
                raise RegistryConflict("active bundle changed; stale activation/rollback rejected")
            if action == "rollback":
                prior = db.execute(
                    "SELECT previous_bundle FROM transitions WHERE model_name=? ORDER BY seq DESC LIMIT 1",
                    (name,),
                ).fetchone()
                if not prior or prior[0] is None:
                    raise RegistryConflict("no previous active bundle to restore")
                bundle_id = prior[0]
            bundle = self._bundle(db, bundle_id)
            if bundle["model_name"] != name:
                raise RegistryConflict("bundle belongs to another model name")
            if current == bundle_id:
                return bundle_id
            previous = db.execute(
                "SELECT seq,sha256 FROM transitions ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            event = dict(
                seq=previous["seq"] + 1 if previous else 1,
                model_name=name,
                previous_bundle=current,
                bundle_id=bundle_id,
                action=action,
                recorded_at=datetime.now(timezone.utc).isoformat(),
                previous_hash=previous["sha256"] if previous else "0" * 64,
            )
            db.execute(
                "INSERT INTO active VALUES (?,?) ON CONFLICT(model_name) DO UPDATE SET bundle_id=excluded.bundle_id",
                (name, bundle_id),
            )
            self._checkpoint("pointer_before_event")
            db.execute(
                "INSERT INTO transitions VALUES (:seq,:model_name,:previous_bundle,:bundle_id,:action,:recorded_at,:previous_hash,:sha256)",
                event | dict(sha256=sha(encode(event))),
            )
            self._checkpoint("transition_before_commit")
        return bundle_id

    def active_bundle(self, name):
        with self.connection() as db:
            self._verify(db)
            active = db.execute(
                "SELECT bundle_id FROM active WHERE model_name=?", (name,)
            ).fetchone()
            if active is None:
                raise KeyError("no active diagnostic bundle")
            return self._bundle(db, active[0])

    def _verify(self, db):
        if (
            db.execute("PRAGMA application_id").fetchone()[0] != APP_ID
            or db.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise RegistryIntegrityError("registry schema identity changed")
        if (
            db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or db.execute("PRAGMA foreign_key_check").fetchone()
        ):
            raise RegistryIntegrityError("SQLite registry integrity failure")
        required = {
            f"immutable_{table}_{action}"
            for table in ("blobs", "bundles", "transitions", "legacy_records")
            for action in ("UPDATE", "DELETE")
        }
        triggers = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        if not required <= triggers:
            raise RegistryIntegrityError("registry immutability triggers missing")
        for blob in db.execute("SELECT * FROM blobs"):
            if sha(blob["content"]) != blob["sha256"]:
                raise RegistryIntegrityError("artifact bytes modified")
        for row in db.execute("SELECT bundle_id FROM bundles"):
            self._bundle(db, row[0])
        previous, state, prior_targets, count = "0" * 64, {}, {}, 0
        for row in db.execute("SELECT * FROM transitions ORDER BY seq"):
            event = dict(row)
            digest = event.pop("sha256")
            count += 1
            if (
                event["seq"] != count
                or event["previous_hash"] != previous
                or sha(encode(event)) != digest
                or event["previous_bundle"] != state.get(event["model_name"])
                or event["action"] not in ("activate", "rollback")
            ):
                raise RegistryIntegrityError("registry transition chain mismatch")
            if event["action"] == "rollback" and event["bundle_id"] != prior_targets.get(
                event["model_name"]
            ):
                raise RegistryIntegrityError("rollback target differs from prior active bundle")
            prior_targets[event["model_name"]] = event["previous_bundle"]
            state[event["model_name"]] = event["bundle_id"]
            previous = digest
        active = {r["model_name"]: r["bundle_id"] for r in db.execute("SELECT * FROM active")}
        if active != state:
            raise RegistryIntegrityError("active pointer differs from committed history")
        for row in db.execute("SELECT * FROM legacy_records"):
            if json.loads(row["payload"])["model_id"] != row["model_id"]:
                raise RegistryIntegrityError("quarantined legacy record identity mismatch")
        snapshot = {
            table: [dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY 1,2")]
            for table in ("bundles", "active", "transitions", "legacy_records")
        }
        snapshot["blob_hashes"] = [
            r[0] for r in db.execute("SELECT sha256 FROM blobs ORDER BY sha256")
        ]
        return dict(
            bundles=len(snapshot["bundles"]),
            transitions=count,
            active=active,
            legacy_records=len(snapshot["legacy_records"]),
            state_sha256=sha(encode(snapshot)),
            scope="local diagnostic registry; not release approval or independent custody",
        )

    def verify(self):
        with self.connection() as db:
            return self._verify(db)

    def import_legacy(self, source):
        source = Path(source).resolve()
        if source == self.path or Path(str(source) + "-wal").exists():
            raise ValueError("legacy migration requires a separate, quiescent DB without WAL")
        before = sha(source.read_bytes())
        uri = "file:" + quote(source.as_posix(), safe="/:") + "?mode=ro"
        legacy = sqlite3.connect(uri, uri=True)
        legacy.row_factory = sqlite3.Row
        try:
            legacy.execute("BEGIN")
            rows = [
                dict(r) for r in legacy.execute("SELECT * FROM model_registry ORDER BY model_id")
            ]
            if sha(source.read_bytes()) != before or Path(str(source) + "-wal").exists():
                raise RegistryConflict("legacy source changed during read-only migration")
            with self.connection(write=True) as db:
                self._verify(db)
                for row in rows:
                    payload = encode(row).decode()
                    existing = db.execute(
                        "SELECT payload FROM legacy_records WHERE source_sha256=? AND model_id=?",
                        (before, row["model_id"]),
                    ).fetchone()
                    if existing is not None and existing[0] != payload:
                        raise RegistryIntegrityError(
                            "quarantined source record differs from repeated import"
                        )
                    db.execute(
                        "INSERT INTO legacy_records VALUES (?,?,?) ON CONFLICT(source_sha256,model_id) DO NOTHING",
                        (before, row["model_id"], payload),
                    )
                self._checkpoint("migration_before_commit")
        finally:
            legacy.close()
        return dict(
            rows=len(rows),
            source_sha256=before,
            activated=0,
            status="quarantined historical records; artifact identity not established",
        )

    def backup(self, target):
        return self._copy_snapshot(self.path, target)

    @classmethod
    def restore(cls, source, target, *, expected_sha256):
        source = Path(source)
        if sha(source.read_bytes()) != expected_sha256 or Path(str(source) + "-wal").exists():
            raise RegistryIntegrityError("backup checksum mismatch")
        result = cls._copy_snapshot(source, target, expected_source_sha256=expected_sha256)
        return cls(target), result

    @classmethod
    def _copy_snapshot(cls, source, target, expected_source_sha256=None):
        source, target = Path(source).resolve(), Path(target).resolve()
        if target.exists():
            raise FileExistsError("backup/restore never overwrites an existing target")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Work beside destination for same-filesystem atomic exclusive publication.
        handle, name = tempfile.mkstemp(prefix=".registry-", suffix=".sqlite", dir=target.parent)
        os.close(handle)
        temporary = Path(name)
        try:
            registry = cls(source)
            with registry.connection() as db:
                expected = registry._verify(db)
                destination = sqlite3.connect(temporary)
                try:
                    db.backup(destination)
                    destination.execute("PRAGMA journal_mode=DELETE")
                finally:
                    destination.close()
            actual = cls(temporary).verify()
            if actual["state_sha256"] != expected["state_sha256"]:
                raise RegistryIntegrityError("snapshot verification mismatch before publication")
            if expected_source_sha256 is not None and (
                sha(source.read_bytes()) != expected_source_sha256
                or Path(str(source) + "-wal").exists()
            ):
                raise RegistryIntegrityError("backup source changed during restore")
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, target)  # Atomic create; fails if another caller wins.
            return dict(sha256=sha(target.read_bytes()), verified_state=actual)
        finally:
            temporary.unlink(missing_ok=True)
