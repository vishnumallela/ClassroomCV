"""replace_detections atomicity: DELETE + all COPY batches in ONE transaction.

Regression for a torn-write defect: DELETE and each COPY batch used to
autocommit independently, so a mid-write failure permanently destroyed the
previous detection set and left a committed partial prefix that /rederive
would trust. Uses a fake asyncpg connection (no DB needed) that records
whether every statement ran inside an open transaction and whether the
transaction committed or rolled back.
"""

import pytest

from app import db
from app.models import Detection


def _det(ts: int) -> Detection:
    return Detection(
        video_ts_ms=ts,
        raw_track_id=1,
        bbox={"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2},
        conf=0.9,
        standing=False,
        back_to_camera=False,
        track_no=1,
    )


class _FakeTransaction:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    async def __aenter__(self):
        self._conn.in_tx = True
        self._conn.log.append("BEGIN")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._conn.log.append("ROLLBACK" if exc_type else "COMMIT")
        self._conn.in_tx = False
        return False


class _FakeConn:
    def __init__(
        self,
        fail_on_copy: int | None = None,
        video_exists: bool = True,
        workflow_run_id: str | None = None,
    ) -> None:
        self.log: list = []
        self.in_tx = False
        self._copies = 0
        self._fail_on_copy = fail_on_copy
        self._video_exists = video_exists
        self._workflow_run_id = workflow_run_id

    def transaction(self):
        return _FakeTransaction(self)

    async def fetchrow(self, sql, *args):
        assert self.in_tx, f"statement ran outside a transaction: {sql}"
        assert "FOR SHARE" in sql, "video-exists check must lock the row"
        self.log.append("SELECT videos")
        if not self._video_exists:
            return None
        return {"workflow_run_id": self._workflow_run_id}

    async def execute(self, sql, *args):
        assert self.in_tx, f"statement ran outside a transaction: {sql}"
        self.log.append("DELETE")

    async def copy_records_to_table(self, table, records=None, columns=None):
        assert self.in_tx, "COPY ran outside a transaction"
        self._copies += 1
        if self._fail_on_copy == self._copies:
            raise ConnectionError("connection dropped mid-COPY")
        self.log.append(("COPY", len(records)))

    async def close(self):
        self.log.append("CLOSE")


async def test_replace_detections_wraps_delete_and_copies_in_one_transaction(
    monkeypatch,
):
    conn = _FakeConn()

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    n = await db.replace_detections("vid", [_det(i) for i in range(5)], batch_size=2)
    assert n == 5
    assert conn.log == [
        "BEGIN",
        "SELECT videos",
        "DELETE",
        ("COPY", 2),
        ("COPY", 2),
        ("COPY", 1),
        "COMMIT",
        "CLOSE",
    ]


async def test_replace_detections_rolls_back_on_mid_copy_failure(monkeypatch):
    conn = _FakeConn(fail_on_copy=2)

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    with pytest.raises(ConnectionError):
        await db.replace_detections("vid", [_det(i) for i in range(5)], batch_size=2)
    # the torn write rolled back instead of committing a partial prefix,
    # and the connection was still closed
    assert conn.log[-2:] == ["ROLLBACK", "CLOSE"]
    assert "COMMIT" not in conn.log


async def test_replace_detections_aborts_when_video_deleted(monkeypatch):
    """Orphan-write fence: detection_events has no FK to videos, so if the
    video was deleted mid-analysis the writer must raise VideoDeletedError
    inside the transaction (rolling back, writing nothing) instead of
    committing permanently orphaned rows."""
    conn = _FakeConn(video_exists=False)

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    with pytest.raises(db.VideoDeletedError):
        await db.replace_detections("vid", [_det(i) for i in range(5)], batch_size=2)
    # nothing was deleted or copied, the transaction rolled back, conn closed
    assert "DELETE" not in conn.log
    assert not any(isinstance(entry, tuple) for entry in conn.log)  # no COPY
    assert conn.log == ["BEGIN", "SELECT videos", "ROLLBACK", "CLOSE"]


async def test_replace_detections_aborts_when_run_superseded(monkeypatch):
    """Stale-run fence: a superseded YOLO job (videos.workflow_run_id was
    re-pointed by a newer reanalyze) must roll back instead of rewriting
    detection_events — the root of the 'done with 376k detections but zero
    tracks' inconsistency."""
    conn = _FakeConn(workflow_run_id="run-NEW")

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    with pytest.raises(db.StaleRunError):
        await db.replace_detections(
            "vid",
            [_det(i) for i in range(5)],
            batch_size=2,
            run_tokens=["attempt-OLD", "run-OLD"],
        )
    assert "DELETE" not in conn.log
    assert conn.log == ["BEGIN", "SELECT videos", "ROLLBACK", "CLOSE"]


async def test_replace_detections_accepts_matching_or_null_token(monkeypatch):
    """The fence accepts the run's own tokens and a NULL stored value (fresh
    upload before the route persists the run id)."""
    for stored in ("attempt-A", "run-A", None):
        conn = _FakeConn(workflow_run_id=stored)

        async def fake_connect(dsn=None, conn=conn):
            return conn

        monkeypatch.setattr(db, "_connect", fake_connect)
        n = await db.replace_detections(
            "vid",
            [_det(i) for i in range(3)],
            batch_size=2,
            run_tokens=["attempt-A", "run-A"],
        )
        assert n == 3
        assert conn.log[-2:] == ["COMMIT", "CLOSE"], f"stored={stored!r}"


async def test_replace_detections_skips_token_check_without_tokens(monkeypatch):
    """No run_tokens (tests, /rederive, direct API use) -> only the
    video-exists check applies, whatever workflow_run_id holds."""
    conn = _FakeConn(workflow_run_id="someone-elses-run")

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    n = await db.replace_detections("vid", [_det(i) for i in range(2)], batch_size=2)
    assert n == 2
    assert conn.log[-2:] == ["COMMIT", "CLOSE"]
