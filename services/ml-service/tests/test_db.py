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
from app.models import (
    CLASS_DOOR,
    CLASS_POINTING,
    CLASS_SCREEN,
    CLASS_TEACHER,
    CLASS_WRITING,
    Detection,
)


def _det(ts: int, cls: int = CLASS_TEACHER, conf: float = 0.9, track_no=1) -> Detection:
    return Detection(
        video_ts_ms=ts,
        cls=cls,
        bbox={"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2},
        conf=conf,
        track_no=track_no,
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
        self.copied: list = []
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
        self.copied.extend(records)
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


# --------------------------------------------------------------------------- #
# What gets stored (migration 0014)
# --------------------------------------------------------------------------- #
#
# The write filter is the whole of Phase 1, and it is one line that is easy to
# get subtly wrong in two opposite directions: too tight and the evidence of a
# second adult is destroyed before the database sees it (the bug), too loose and
# "only the teacher class is ever stored" quietly stops being true (a privacy
# property, not a preference). Both directions are pinned below.


def _copied_columns(conn) -> list[tuple]:
    """(video_ts_ms, track_no, cls) for every row the fake connection COPYed."""
    import json as _json

    return [(r[0], r[2], _json.loads(r[5])["cls"]) for r in conn.copied]


async def _write(monkeypatch, detections, conn=None):
    conn = conn or _FakeConn()

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    n = await db.replace_detections("vid", detections)
    return conn, n


async def test_unattributed_teacher_boxes_are_persisted(monkeypatch):
    """The Phase 1 point: a teacher box no track claimed is EVIDENCE, not noise.

    On the lesson that prompted this, 285 of 289 co-present instants were never
    even recorded as rejected — the second adult simply vanished on the way to
    the database, which is why a stored-row replay could never rediscover her.
    """
    dets = [_det(0), _det(0, track_no=None), _det(200), _det(200, track_no=None)]
    conn, n = await _write(monkeypatch, dets)

    assert n == 4
    assert _copied_columns(conn) == [
        (0, 1, CLASS_TEACHER),
        (0, None, CLASS_TEACHER),
        (200, 1, CLASS_TEACHER),
        (200, None, CLASS_TEACHER),
    ]


async def test_only_the_teacher_class_is_ever_stored(monkeypatch):
    """Regression guard for the tempting one-character fix.

    Before 0014, `track_no is not None` was ALSO what kept every other class
    out, because only teacher boxes were ever stamped. Loosening that check
    instead of testing the class would have started persisting screen, door,
    pointing and writing rows — silently, with no test failing, and with the
    privacy claim in this module's docstring no longer true of the data.
    """
    dets = [
        _det(0, cls=CLASS_TEACHER, track_no=None),
        _det(0, cls=CLASS_SCREEN, track_no=None),
        _det(0, cls=CLASS_DOOR, track_no=None),
        _det(0, cls=CLASS_POINTING, track_no=None),
        _det(0, cls=CLASS_WRITING, track_no=None),
    ]
    conn, n = await _write(monkeypatch, dets)

    assert n == 1
    assert {cls for _, _, cls in _copied_columns(conn)} == {CLASS_TEACHER}


async def test_sub_threshold_teacher_boxes_are_not_stored(monkeypatch):
    """The detector emits down to a low floor so one pass serves every
    consumer. Storing that floor would store noise no attribution rule would
    ever consult, so the write applies the same teacher_conf the tracker uses
    to decide what counts as a candidate."""
    from app.config import get_settings

    threshold = get_settings().teacher_conf
    dets = [
        _det(0, conf=threshold + 0.1, track_no=None),
        _det(200, conf=threshold - 0.1, track_no=None),
    ]
    conn, n = await _write(monkeypatch, dets)

    assert n == 1
    assert [ts for ts, _, _ in _copied_columns(conn)] == [0]


async def test_fetch_reads_an_unattributed_row_back_as_none(monkeypatch):
    """The round trip that makes /rederive able to re-ask the attribution
    question rather than replay its old answer: the losing candidates come back
    in the input, shaped exactly as a fresh detector pass would hand them over.
    """
    rows = [
        {"video_ts_ms": 0, "track_no": 1, "bbox": {"x": 0.1, "y": 0.1, "w": 0.1,
         "h": 0.2}, "confidence": 0.9, "meta": {"cls": CLASS_TEACHER}},
        {"video_ts_ms": 0, "track_no": None, "bbox": {"x": 0.5, "y": 0.1,
         "w": 0.1, "h": 0.2}, "confidence": 0.8, "meta": {"cls": CLASS_TEACHER}},
    ]

    class _ReadConn:
        async def fetch(self, sql, *args):
            return rows

        async def close(self):
            pass

    async def fake_connect(dsn=None):
        return _ReadConn()

    monkeypatch.setattr(db, "_connect", fake_connect)
    out = await db.fetch_detections("vid")

    assert [d.track_no for d in out] == [1, None]
    assert all(d.cls == CLASS_TEACHER for d in out)


async def test_appearance_descriptor_round_trips_through_meta(monkeypatch):
    """Phase 3 stores the descriptor beside the class id in meta (jsonb,
    additive) so attribution can be re-derived from rows alone."""
    import json

    captured: list = []

    class _Conn(_FakeConn):
        async def copy_records_to_table(self, table, records=None, columns=None):
            captured.extend(records)
            await super().copy_records_to_table(table, records, columns)

    conn = _Conn()

    async def fake_connect(dsn=None):
        return conn

    monkeypatch.setattr(db, "_connect", fake_connect)
    with_app = _det(0)
    with_app.app = [0.1] * 16
    without = _det(200)
    await db.replace_detections("vid", [with_app, without])
    metas = [json.loads(r[5]) for r in captured]
    assert metas[0]["app"] == [0.1] * 16 and metas[0]["cls"] == CLASS_TEACHER
    assert "app" not in metas[1], "no descriptor -> no key, not a null"
