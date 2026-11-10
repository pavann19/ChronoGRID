"""Shared isolation fixtures for tests that exercise persistent subsystems."""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from chronogrid.db.models import Base


@pytest.fixture(scope="session", autouse=True)
def isolate_persisted_runtime_state(tmp_path_factory):
    """Keep service state written by tests outside the working tree."""
    state_root = tmp_path_factory.mktemp("chronogrid-test-runtime-state")
    keys = ("DDIR_STATE_FILE", "DQCP_STATE_FILE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["DDIR_STATE_FILE"] = str(state_root / "ddir_runtime_state.json")
    os.environ["DQCP_STATE_FILE"] = str(state_root / "dqcp_aci_state.json")
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def isolated_application_database(monkeypatch):
    """Route persistence tests to a disposable in-memory SQLite database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    @contextmanager
    def get_isolated_session(db_url: str | None = None):
        del db_url
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("chronogrid.retraining.promotion_engine.get_session", get_isolated_session)
    monkeypatch.setattr("chronogrid.monitoring.drift_detector.get_session", get_isolated_session)
    try:
        yield get_isolated_session
    finally:
        engine.dispose()
