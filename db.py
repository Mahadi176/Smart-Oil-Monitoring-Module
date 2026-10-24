"""SQLite persistence and authentication for Smart Oil Monitor."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

DB_PATH = Path(os.environ.get("OIL_MONITOR_DB", "oil_monitor.db"))
PBKDF2_ROUNDS = 310_000


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                usage_date TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('Household','Vehicle','Industrial')),
                liters REAL NOT NULL CHECK(liters >= 0),
                cost REAL NOT NULL CHECK(cost >= 0),
                note TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'manual' CHECK(source IN ('manual','demo','sensor_import')),
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage(user_id, usage_date);
            CREATE INDEX IF NOT EXISTS idx_usage_user_category_date ON usage(user_id, category, usage_date);
            """
        )


def _password_digest(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ROUNDS)
    return digest.hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not 3 <= len(username) <= 40 or not all(c.isalnum() or c in "_.-" for c in username):
        return False, "Username must be 3–40 characters and use letters, numbers, dots, underscores, or hyphens."
    if len(password) < 10:
        return False, "Password must contain at least 10 characters."
    salt = secrets.token_hex(16)
    digest = _password_digest(password, salt)
    try:
        with connection() as conn:
            conn.execute(
                "INSERT INTO users(username,password_hash,password_salt,created_at) VALUES(?,?,?,?)",
                (username, digest, salt, datetime.utcnow().isoformat(timespec="seconds")),
            )
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username is already registered."


def authenticate(username: str, password: str) -> int | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT id,password_hash,password_salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    candidate = _password_digest(password, row["password_salt"])
    return int(row["id"]) if hmac.compare_digest(candidate, row["password_hash"]) else None


def get_username(user_id: int) -> str:
    with connection() as conn:
        row = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    if row is None:
        raise ValueError("Unknown user.")
    return str(row["username"])


def insert_usage(user_id: int, usage_date: date, category: str, liters: float, cost: float, note: str = "", source: str = "manual") -> int:
    if category not in {"Household", "Vehicle", "Industrial"}:
        raise ValueError("Invalid category.")
    if liters < 0 or cost < 0:
        raise ValueError("Liters and cost must be non-negative.")
    if liters > 1_000_000 or cost > 1_000_000_000:
        raise ValueError("Value exceeds the configured safety limit; verify the entry.")
    if source not in {"manual", "demo", "sensor_import"}:
        raise ValueError("Invalid source.")
    with connection() as conn:
        cursor = conn.execute(
            "INSERT INTO usage(user_id,usage_date,category,liters,cost,note,source,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (user_id, usage_date.isoformat(), category, float(liters), float(cost), note.strip()[:500], source,
             datetime.utcnow().isoformat(timespec="seconds")),
        )
        return int(cursor.lastrowid)


def delete_usage(user_id: int, entry_id: int) -> bool:
    with connection() as conn:
        cursor = conn.execute("DELETE FROM usage WHERE id=? AND user_id=?", (entry_id, user_id))
        return cursor.rowcount == 1


def load_usage(user_id: int):
    import pandas as pd
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, usage_date AS date, category, liters, cost, note, source, created_at "
            "FROM usage WHERE user_id=? ORDER BY usage_date, id", (user_id,)
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=["id", "date", "category", "liters", "cost", "note", "source", "created_at"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["liters"] = pd.to_numeric(df["liters"], errors="coerce")
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    return df


def seed_demo_data(user_id: int) -> int:
    """Insert 90 days of visibly labelled synthetic data only when account has no entries."""
    import numpy as np
    with connection() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM usage WHERE user_id=?", (user_id,)).fetchone()[0]
    if existing:
        return 0
    rng = np.random.default_rng(42)
    start = date.today() - timedelta(days=89)
    rows = []
    for category, base in [("Household", 3.0), ("Vehicle", 8.0), ("Industrial", 25.0)]:
        for i in range(90):
            d = start + timedelta(days=i)
            seasonal = 1 + 0.10 * np.sin(i * 2 * np.pi / 7)
            liters = max(0.1, base * seasonal * (1 + rng.normal(0, 0.06)))
            rows.append((user_id, d.isoformat(), category, round(float(liters), 2), round(float(liters * 1.35), 2),
                         "Synthetic demonstration record — not measured data", "demo", datetime.utcnow().isoformat(timespec="seconds")))
    with connection() as conn:
        conn.executemany(
            "INSERT INTO usage(user_id,usage_date,category,liters,cost,note,source,created_at) VALUES(?,?,?,?,?,?,?,?)", rows
        )
    return len(rows)
