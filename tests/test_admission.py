import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from paper_evidence_rag.admission import Admission, AdmissionDenied


def test_restart_keeps_counts_and_no_raw_identity(tmp_path):
    path = tmp_path / "budget.sqlite"
    first = Admission(path, initialize=True, daily=2, visitor_daily=1)
    with first.admit("private-session") as reservation:
        reservation.commit()
    reopened = Admission(path, daily=2, visitor_daily=1)
    with pytest.raises(AdmissionDenied, match="question allowance"):
        with reopened.admit("private-session"):
            pass
    with reopened.admit("second-session") as reservation:
        reservation.commit()
    with pytest.raises(AdmissionDenied, match="current processing capacity"):
        with reopened.admit("rotated-session"):
            pass
    assert b"private-session" not in path.read_bytes()


def test_failed_work_releases_slot_and_concurrency(tmp_path):
    limit = Admission(tmp_path / "budget.sqlite", initialize=True, visitor_daily=1)
    with pytest.raises(ValueError):
        with limit.admit("one"):
            raise ValueError("work failed")
    with limit.admit("two"):
        with pytest.raises(AdmissionDenied, match="another request"):
            with limit.admit("three"):
                pass
    with limit.admit("one") as reservation:
        reservation.commit()


def test_pending_reservation_expires_after_crash_lease(tmp_path):
    now = [100000.0]
    path = tmp_path / "budget.sqlite"
    first = Admission(path, initialize=True, daily=8, visitor_daily=2,
                      cooldown=5, clock=lambda: now[0])
    reservation = first.admit("one")
    context = reservation.__enter__()
    now[0] += 3601
    reopened = Admission(path, daily=8, visitor_daily=2,
                         cooldown=5, clock=lambda: now[0])
    with reopened.admit("two") as committed:
        committed.commit()
    reservation.__exit__(None, None, None)


def test_committed_records_remain_after_restart(tmp_path):
    path = tmp_path / "budget.sqlite"
    first = Admission(path, initialize=True, daily=1, visitor_daily=1)
    with first.admit("one") as reservation:
        reservation.commit()

    reopened = Admission(path, daily=1, visitor_daily=1)
    with pytest.raises(AdmissionDenied, match="current processing capacity"):
        with reopened.admit("two"):
            pass


def test_lost_database_fails_closed(tmp_path):
    path = tmp_path / "budget.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        Admission(path)
    limit = Admission(path, initialize=True)
    path.unlink()
    with pytest.raises(AdmissionDenied, match="unavailable"):
        with limit.admit("one"):
            pass
    assert not path.exists()


def test_atomic_global_reservations(tmp_path):
    path = tmp_path / "budget.sqlite"
    Admission(path, initialize=True)
    def run(index):
        limit = Admission(path, daily=2, visitor_daily=1)
        try:
            with limit.admit(str(index)) as reservation:
                reservation.commit()
            return True
        except AdmissionDenied:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(run, range(16))) <= 2
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM admissions").fetchone()[0] <= 2


def test_pdf_processing_has_separate_higher_shared_capacity(tmp_path):
    limit = Admission(tmp_path / "budget.sqlite", initialize=True, daily=1,
                      visitor_daily=1, upload_daily=3, upload_visitor_daily=2)
    with limit.admit("query-user", "query") as reservation:
        reservation.commit()
    with pytest.raises(AdmissionDenied, match="current processing capacity"):
        with limit.admit("another-query", "query"):
            pass
    for visitor in ("upload-one", "upload-two", "upload-three"):
        with limit.admit(visitor, "upload") as reservation:
            reservation.commit()
    with pytest.raises(AdmissionDenied, match="current processing capacity"):
        with limit.admit("upload-four", "upload"):
            pass


def test_rolling_window_cooldown_and_separate_upload_budget(tmp_path):
    now = [100000.0]
    limit = Admission(tmp_path / "budget.sqlite", initialize=True, daily=2,
                      visitor_daily=2, clock=lambda: now[0])
    with limit.admit("one") as reservation:
        reservation.commit()
    with pytest.raises(AdmissionDenied, match="wait"):
        with limit.admit("one"):
            pass
    with limit.admit("one", "upload") as reservation:
        reservation.commit()
    now[0] += 86401
    with limit.admit("one"):
        pass


@pytest.mark.parametrize(("kind", "label"), [
    ("query", "question"),
    ("upload", "PDF processing"),
])
def test_session_denial_reports_operation_kind_and_visitor_cap(tmp_path, kind, label):
    now = [0]
    limit = Admission(tmp_path / f"{kind}.sqlite", initialize=True,
                      daily=10, visitor_daily=2, cooldown=5, clock=lambda: now[0])
    for _ in range(2):
        now[0] += 6
        with limit.admit("same-session", kind) as reservation:
            reservation.commit()
    now[0] += 6

    with pytest.raises(AdmissionDenied) as caught:
        with limit.admit("same-session", kind):
            pytest.fail("The configured per-session cap should deny this attempt")

    message = str(caught.value)
    assert message.startswith("No ") and "allowance remains" in message
    assert label in message
    assert "limit" not in message
