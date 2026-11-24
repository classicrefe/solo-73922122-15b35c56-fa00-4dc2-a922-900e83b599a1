"""只读快照事务的测试：修订号、定位点与分析结果取自同一次已提交状态。

详情页/API 的一次取数在同一个 SQLite 只读事务（WAL 快照隔离）里完成：

* 取数期间其他访问者补点、改点、撤点不会被阻塞，且本次页面不会读到半新半旧；
* 若取数开始时晶圆已被删除，返回正常的 404 未找到页；
* API 返回的修订号与定位点、分析结果属于同一个快照。
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

from wafer_review.app import create_app
from wafer_review.db import connect, init_db, snapshot
from wafer_review.model import analyze
from wafer_review.service import (
    StepInput,
    correct_locator,
    create_wafer,
    delete_wafer,
    load_wafer,
    remove_locator,
)

SAMPLES = [
    ("光刻", "1_光刻.csv"),
    ("刻蚀", "2_刻蚀.csv"),
    ("沉积", "3_沉积.csv"),
    ("研磨", "4_研磨.csv"),
    ("检测", "5_检测.csv"),
    ("封装", "6_封装.csv"),
]
SAMPLE_DIR = Path("/workspace/samples")


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "snap.sqlite3")
    init_db(path)
    return path


@pytest.fixture()
def app(db_path):
    return create_app({"DATABASE": db_path, "TESTING": True})


@pytest.fixture()
def client(app):
    return app.test_client()


def _create_demo(db_path: str) -> int:
    steps = [
        StepInput(name, (SAMPLE_DIR / filename).read_text(encoding="utf-8"))
        for name, filename in SAMPLES
    ]
    return create_wafer(db_path, "SNAP", 8, steps)


def test_snapshot_holds_revision_while_concurrent_writer_commits(db_path):
    wafer_id = _create_demo(db_path)
    barrier = threading.Event()
    seen: dict[str, object] = {}

    def reader() -> None:
        with snapshot(db_path) as conn:
            wafer = load_wafer(conn, wafer_id)
            assert wafer is not None
            barrier.set()  # 快照已开始，通知写线程动手
            # 等写线程提交完成（给足时间但不依赖 sleep 时序正确性）
            barrier.wait(5)
            threading.Event().wait(0.4)
            # 同一事务内重新读取：修订号与定位点必须仍是旧快照
            same = load_wafer(conn, wafer_id)
            seen["revision"] = same.revision
            seen["c1_rows"] = [
                (loc.pid, loc.row, loc.col)
                for step in same.steps
                for loc in step.locators
                if loc.pid == "C1"
            ]
            seen["scheme_count"] = analyze(same).feasibility.scheme_count

    t = threading.Thread(target=reader)
    t.start()
    assert barrier.wait(5)

    # 写线程在读者持快照期间撤去沉积 C1（新修订：方案数 2 -> 8）
    with connect(db_path) as conn:
        wafer = load_wafer(conn, wafer_id)
    c1 = next(loc for loc in wafer.steps[2].locators if loc.pid == "C1")
    remove_locator(db_path, wafer_id, c1.id)

    t.join(5)
    assert not t.is_alive()
    # 读者拿到的是已提交的旧状态（r0、2 套方案、C1 仍在）
    assert seen["revision"] == 0
    assert seen["scheme_count"] == 2
    assert ("C1", 4, 5) in seen["c1_rows"]

    # 新连接看到的是新状态
    with connect(db_path) as conn:
        refreshed = load_wafer(conn, wafer_id)
    assert refreshed.revision == 1
    assert analyze(refreshed).feasibility.scheme_count == 8


def test_concurrent_writes_are_not_blocked_by_snapshot(db_path):
    wafer_id = _create_demo(db_path)
    entered = threading.Event()
    release = threading.Event()
    write_errors: list[Exception] = []

    def long_reader() -> None:
        with snapshot(db_path) as conn:
            load_wafer(conn, wafer_id)
            entered.set()
            release.wait(5)

    reader = threading.Thread(target=long_reader)
    reader.start()
    assert entered.wait(5)

    def writer() -> None:
        try:
            with connect(db_path) as conn:
                wafer = load_wafer(conn, wafer_id)
            c1 = next(loc for loc in wafer.steps[5].locators if loc.pid == "C1")
            correct_locator(db_path, wafer_id, c1.id, 4, 4)
        except Exception as exc:  # pragma: no cover - 失败时留证据
            write_errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    t.join(10)
    release.set()
    reader.join(5)
    assert write_errors == []
    with connect(db_path) as conn:
        assert load_wafer(conn, wafer_id).revision == 1


def test_detail_page_renders_consistent_revision_during_concurrent_change(app, client, db_path):
    wafer_id = _create_demo(db_path)

    # 详情页 200 且页面展示的修订号与定位点接口一致（同一已提交状态）
    page = client.get(f"/wafers/{wafer_id}?step=6&did=D102")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "修订 r0" in body
    locators = client.get(f"/api/wafers/{wafer_id}/locators").get_json()
    defects = client.get(f"/api/wafers/{wafer_id}/defects").get_json()
    assert locators["revision"] == 0
    assert defects["revision"] == 0
    assert defects["scheme_count"] == 2

    # 并发改点后再取：页面整体切到新修订，不会出现旧修订号配新结果
    with connect(db_path) as conn:
        wafer = load_wafer(conn, wafer_id)
    c1 = next(loc for loc in wafer.steps[5].locators if loc.pid == "C1")
    correct_locator(db_path, wafer_id, c1.id, 4, 4)
    page2 = client.get(f"/wafers/{wafer_id}?step=6&did=D104")
    body2 = page2.get_data(as_text=True)
    assert "修订 r1" in body2
    payload = client.get(f"/api/wafers/{wafer_id}/defects").get_json()
    assert payload["revision"] == 1
    packing = {r["defect"]: r for r in payload["results"] if r["step"] == "封装"}
    assert packing["D104"]["coordinate"] == [5, 5]


def test_delete_during_fetch_returns_404_or_full_old_record(db_path):
    wafer_id = _create_demo(db_path)

    # 场景 A：快照开始后删除 —— 旧快照仍能读全记录
    with snapshot(db_path) as conn:
        wafer = load_wafer(conn, wafer_id)
        assert wafer is not None and len(wafer.steps) == 6
        delete_wafer(db_path, wafer_id)
        old = load_wafer(conn, wafer_id)
        assert old is not None
        assert old.revision == wafer.revision
        assert len(old.steps) == 6
        # 旧快照上的分析也完整可用
        assert analyze(old).feasibility.scheme_count == 2

    # 场景 B：删除后新开的取数 —— 正常未找到
    with snapshot(db_path) as conn:
        assert load_wafer(conn, wafer_id) is None

    app = create_app({"DATABASE": db_path, "TESTING": True})
    client = app.test_client()
    assert client.get(f"/wafers/{wafer_id}").status_code == 404
    assert client.get(f"/api/wafers/{wafer_id}/defects").status_code == 404
    not_found = client.get(f"/wafers/{wafer_id}").get_data(as_text=True)
    assert "没有这条记录" in not_found


def test_upload_then_detail_uses_single_snapshot(app, client):
    data = {"name": "SNAP-HTTP", "grid_size": "8"}
    for i, (name, filename) in enumerate(SAMPLES, start=1):
        data[f"name_{i}"] = name
        data[f"csv_{i}"] = (
            io.BytesIO((SAMPLE_DIR / filename).read_bytes()),
            filename,
        )
    resp = client.post("/wafers", data=data, content_type="multipart/form-data")
    assert resp.status_code == 302
    page = client.get("/wafers/1?step=6")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    # 全局绘图方案在页面上只标注一套，且叠加/换算说明已更新
    assert "绘图采用" in body
    assert "同一套全局可行方案" in body
