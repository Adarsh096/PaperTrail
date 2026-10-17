"""Filesystem-only contract tests for the private mounted-bucket ledger."""
from contextlib import contextmanager
import json
import hashlib
import threading

import pytest

from paper_evidence_rag import bucket_admission as bucket


@contextmanager
def local_lock_provider(path):
    yield


def noop_directory_sync(path):
    """Test double for the unsupported Windows directory-fsync operation."""


def make_admission(root, **options):
    options.setdefault("mount_checker", lambda path: True)
    options.setdefault("directory_sync", noop_directory_sync)
    options.setdefault("lock_provider", local_lock_provider)
    return bucket.MountedBucketAdmission(root, **options)


@pytest.fixture
def ledger(tmp_path):
    root = tmp_path / "mounted-bucket"
    root.mkdir()
    ledger_id = bucket.initialize_bucket(root, mount_checker=lambda path: True,
                                        directory_sync=noop_directory_sync)
    admission = make_admission(root,
        local_lock_path=tmp_path / "local-lock" / "ledger.lock",
    )
    return root, ledger_id, admission


def test_failed_attempt_is_rolled_back_and_restart_preserves_legacy_commits(ledger, tmp_path):
    root, ledger_id, admission = ledger
    with pytest.raises(RuntimeError, match="provider failed"):
        with admission.admit("private-session-value"):
            raise RuntimeError("provider failed")
    # Version-1 records from the previous ledger format remain committed.
    legacy = bucket._signed({"version": 1, "id": "a" * 32,
        "ledger_id": admission.metadata["ledger_id"],
        "salt_id": hashlib.sha256(bytes.fromhex(admission.metadata["salt"])).hexdigest(),
        "visitor": "b" * 64, "kind": "query", "at": admission.clock()})
    (root / ("a" * 32 + ".json")).write_bytes(bucket._canonical(legacy))
    reopened = make_admission(root,
        local_lock_path=tmp_path / "local-lock" / "ledger.lock",
        query_daily=1, visitor_query_daily=1)
    with pytest.raises(bucket.AdmissionDenied, match="current processing capacity"):
        with reopened.admit("different-session"):
            pytest.fail("A persisted reservation must continue to count after restart")
    assert reopened.status() == {"ledger_id": ledger_id, "reservation_count": 1}
    assert not (root / ("a" * 32 + ".json")).is_symlink()


def test_query_and_upload_allowances_are_separate(ledger, tmp_path):
    root, _, _ = ledger
    admission = make_admission(root,
        local_lock_path=tmp_path / "lock", mount_checker=lambda path: True,
        query_daily=1, visitor_query_daily=1,
        upload_daily=1, visitor_upload_daily=1)
    with admission.admit("query-user", "query") as reservation:
        reservation.commit()
    with admission.admit("upload-user", "upload") as reservation:
        reservation.commit()
    for kind in ("query", "upload"):
        with pytest.raises(bucket.AdmissionDenied, match="current processing capacity"):
            with admission.admit("other-user", kind):
                pytest.fail("Each operation kind has its own global allowance")


@pytest.mark.parametrize(("kind", "label"), [
    ("query", "question"),
    ("upload", "PDF processing"),
])
def test_global_denial_avoids_exposing_shared_cap(ledger, tmp_path, kind, label):
    root, _, _ = ledger
    options = {"query_daily": 3, "visitor_query_daily": 2,
               "upload_daily": 4, "visitor_upload_daily": 2}
    admission = make_admission(root, local_lock_path=tmp_path / "custom-lock", **options)
    limit, _ = admission.limits[kind]

    for index in range(limit):
        with admission.admit(f"{kind}-{index}", kind) as reservation:
            reservation.commit()
    with pytest.raises(bucket.AdmissionDenied) as caught:
        with admission.admit("another-session", kind):
            pytest.fail("The configured shared cap should deny this attempt")

    message = str(caught.value)
    assert "current processing capacity" in message
    assert "shared" not in message
    assert "limit" not in message


@pytest.mark.parametrize(("kind", "label"), [
    ("query", "question"),
    ("upload", "PDF processing"),
])
def test_session_denial_reports_operation_kind_and_visitor_cap(
        ledger, tmp_path, kind, label):
    root, _, _ = ledger
    now = [0]
    options = {"query_daily": 10, "visitor_query_daily": 2,
               "upload_daily": 10, "visitor_upload_daily": 2}
    admission = make_admission(root, local_lock_path=tmp_path / "visitor-lock",
                               cooldown=5, clock=lambda: now[0], **options)
    for _ in range(2):
        now[0] += 6
        with admission.admit("same-session", kind) as reservation:
            reservation.commit()
    now[0] += 6

    with pytest.raises(bucket.AdmissionDenied) as caught:
        with admission.admit("same-session", kind) as reservation:
            reservation.commit()
            pytest.fail("The configured per-session cap should deny this attempt")

    message = str(caught.value)
    assert message.startswith("No ") and "allowance remains" in message
    assert label in message
    assert "limit" not in message


def test_missing_or_corrupt_metadata_fails_closed_without_initializing(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(bucket.AdmissionDenied, match="corrupt"):
        bucket.MountedBucketAdmission(root, mount_checker=lambda path: True)
    assert list(root.iterdir()) == []

    initialized = tmp_path / "corrupt"
    initialized.mkdir()
    bucket.initialize_bucket(initialized, mount_checker=lambda path: True,
                             directory_sync=noop_directory_sync)
    metadata = initialized / "metadata.json"
    metadata.write_text('{"version":1}', encoding="utf-8")
    with pytest.raises(bucket.AdmissionDenied):
        make_admission(initialized)
    assert metadata.read_text(encoding="utf-8") == '{"version":1}'


def test_corrupt_reservation_blocks_before_yield(ledger):
    root, _, admission = ledger
    with admission.admit("first") as reservation:
        reservation.commit()
    reservation = next(root.glob("[0-9a-f]*.json"))
    record = json.loads(reservation.read_text(encoding="utf-8"))
    record["visitor"] = "f" * 64
    reservation.write_text(json.dumps(record), encoding="utf-8")
    yielded = []
    with pytest.raises(bucket.AdmissionDenied, match="corrupt"):
        with admission.admit("second"):
            yielded.append(True)
    assert yielded == []


def test_runtime_metadata_removal_fails_closed_before_yield(ledger):
    root, _, admission = ledger
    (root / "metadata.json").unlink()
    yielded = []
    with pytest.raises(bucket.AdmissionDenied, match="corrupt"):
        with admission.admit("visitor"):
            yielded.append(True)
    assert yielded == []


def test_metadata_replacement_cannot_reset_existing_reservations(ledger, tmp_path):
    root, _, admission = ledger
    with admission.admit("visitor") as reservation:
        reservation.commit()
    metadata_path = root / "metadata.json"
    original = json.loads(metadata_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in original.items() if key != "sha256"}
    unsigned["salt"] = "a" * 64
    metadata_path.write_text(json.dumps(bucket._signed(unsigned)), encoding="utf-8")
    reopened = make_admission(root, local_lock_path=tmp_path / "lock")
    yielded = []
    with pytest.raises(bucket.AdmissionDenied, match="record is invalid"):
        with reopened.admit("visitor"):
            yielded.append(True)
    assert yielded == []


def test_directory_fsync_failure_denies_before_yield(ledger, tmp_path):
    root, _, _ = ledger
    def no_directory_fsync(path):
        raise OSError("directory fsync unsupported")
    admission = make_admission(root, local_lock_path=tmp_path / "lock",
                               directory_sync=no_directory_fsync)
    yielded = []
    with pytest.raises(bucket.AdmissionDenied, match="stored durably"):
        with admission.admit("visitor"):
            yielded.append(True)
    assert yielded == []


def test_mount_must_be_verified_at_startup_and_each_reservation(ledger, tmp_path):
    root, _, _ = ledger
    mounted = [True]
    admission = make_admission(root, local_lock_path=tmp_path / "lock",
                               mount_checker=lambda path: mounted[0])
    mounted[0] = False
    yielded = []
    with pytest.raises(bucket.AdmissionDenied, match="bucket is unavailable"):
        with admission.admit("visitor"):
            yielded.append(True)
    assert yielded == []
    with pytest.raises(ValueError, match="mounted bucket"):
        bucket.initialize_bucket(root, mount_checker=lambda path: False,
                                 directory_sync=noop_directory_sync)


def test_lock_rejects_overlapping_threads(ledger):
    _, _, admission = ledger
    entered, finish = threading.Event(), threading.Event()

    def first_request():
        with admission.admit("first"):
            entered.set()
            assert finish.wait(5)

    first = threading.Thread(target=first_request)
    first.start()
    assert entered.wait(5)
    yielded = []
    with pytest.raises(bucket.AdmissionDenied, match="another request"):
        with admission.admit("second"):
            yielded.append(True)
    finish.set()
    first.join(5)
    assert not first.is_alive()
    assert yielded == []


def test_rolling_expiry_prunes_only_validated_old_reservations(ledger, tmp_path):
    root, _, _ = ledger
    now = [50_000.0]
    admission = make_admission(root,
        local_lock_path=tmp_path / "lock", mount_checker=lambda path: True,
        query_daily=1, visitor_query_daily=1,
        clock=lambda: now[0])
    with admission.admit("old-user") as reservation:
        reservation.commit()
    now[0] += 86_401
    with admission.admit("new-user") as reservation:
        reservation.commit()
    assert admission.status()["reservation_count"] == 1
    assert len(list(root.glob("[0-9a-f]*.json"))) == 1


def test_stale_pending_reservation_expires_after_crash(ledger, tmp_path):
    root, _, _ = ledger
    now = [50_000.0]
    admission = make_admission(root, local_lock_path=tmp_path / "lock",
                               query_daily=1, visitor_query_daily=1,
                               clock=lambda: now[0])
    with admission.admit("crashed"):
        pass  # Uncommitted exit rolls back like a failed operation.
    # Simulate process death by keeping a pending record past its lease.
    record_id = "c" * 32
    pending = bucket._signed({"version": 1, "id": record_id,
        "ledger_id": admission.metadata["ledger_id"],
        "salt_id": hashlib.sha256(bytes.fromhex(admission.metadata["salt"])).hexdigest(),
        "visitor": "d" * 64, "kind": "query", "at": now[0],
        "state": "pending", "expires_at": now[0] + bucket.RESERVATION_TTL})
    (root / (record_id + ".json")).write_bytes(bucket._canonical(pending))
    now[0] += bucket.RESERVATION_TTL + 1
    with admission.admit("new") as reservation:
        reservation.commit()
    assert admission.status()["reservation_count"] == 1
