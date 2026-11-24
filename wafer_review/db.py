"""SQLite 持久化层。

一次“上传 / 更正定位点 / 撤去定位点”都是一个事务：写入新的定位点后，在同一
事务里校验是否存在可行方案。若不存在则回滚，因此“无解或保存失败就沿用原定位
点和原结果”。查询结果不落库，总是由已保存的定位点即时推导，避免结果与定位点
不同步。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS wafer (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    grid_size   INTEGER NOT NULL CHECK (grid_size BETWEEN 2 AND 60),
    revision    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS step (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    wafer_id    INTEGER NOT NULL REFERENCES wafer(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    UNIQUE (wafer_id, seq),
    UNIQUE (wafer_id, name)
);

CREATE TABLE IF NOT EXISTS locator (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id     INTEGER NOT NULL REFERENCES step(id) ON DELETE CASCADE,
    pid         TEXT    NOT NULL,
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    UNIQUE (step_id, pid)
);

CREATE TABLE IF NOT EXISTS defect (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id     INTEGER NOT NULL REFERENCES step(id) ON DELETE CASCADE,
    did         TEXT    NOT NULL,
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    UNIQUE (step_id, did)
);

CREATE INDEX IF NOT EXISTS idx_step_wafer ON step(wafer_id, seq);
CREATE INDEX IF NOT EXISTS idx_locator_step ON locator(step_id);
CREATE INDEX IF NOT EXISTS idx_defect_step ON defect(step_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=15.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def snapshot(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """打开一次只读快照事务。

    ``BEGIN`` 之后的所有 SELECT 都基于同一个已提交版本（WAL 下为快照隔离）：
    即使其他访问者在本事务期间补点、改点、撤点甚至删除整片晶圆，本事务读到的
    修订号、定位点与缺陷仍彼此一致，页面不会用旧修订号显示新结果；写操作也
    不会被阻塞，可以正常保存。事务结束即释放快照，不会长期滞留 WAL。
    """
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.execute("COMMIT")


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
