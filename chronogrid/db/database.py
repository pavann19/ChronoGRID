"""
Database engine and session management for ChronoGrid.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from chronogrid.db.models import Base

_ENGINE = None
_SESSION_FACTORY = None


def get_engine(db_url: str = None):
    """Initializes or returns singleton SQLAlchemy engine."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        return _ENGINE

    if db_url is None:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///data/chronogrid.db")

    if db_url.startswith("sqlite"):
        # Ensure parent directory exists for SQLite
        db_path = db_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _ENGINE = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        _ENGINE = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )

    _SESSION_FACTORY = sessionmaker(autocommit=False, autoflush=False, bind=_ENGINE)
    return _ENGINE


def init_db(db_url: str = None):
    """Creates all database tables."""
    engine = get_engine(db_url)
    Base.metadata.create_all(bind=engine)
    return engine


@contextmanager
def get_session(db_url: str = None) -> Session:
    """Context manager for atomic database sessions."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        get_engine(db_url)

    session = _SESSION_FACTORY()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

