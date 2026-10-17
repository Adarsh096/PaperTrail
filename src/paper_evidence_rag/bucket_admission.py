"""Fail-closed admission ledger for one mounted private bucket and one Space.

This uses immutable local JSON reservations and a host-local lock. Valid records
expire after 24 hours and are retired only after every record has passed its
integrity checks. It provides no distributed lock or multi-replica guarantee.
The mount must support exclusive file creation, file fsync, and directory fsync.
"""
from contextlib import contextmanager
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
import time
import uuid

from .admission import AdmissionDenied

VERSION = 1
MAX_RECORDS = 10_000
RESERVATION_TTL = 3600
_ID = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_LOCKS = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _signed(value):
    return {**value, "sha256": hashlib.sha256(_canonical(value)).hexdigest()}


def _sync_directory(path):
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_new_durable(path, value, directory_sync=_sync_directory):
    payload = _canonical(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_sync(path.parent)


def initialize_bucket(root, *, mount_checker=os.path.ismount, require_mount=True,
                      directory_sync=_sync_directory):
    """Explicit operator setup. Runtime construction never creates metadata."""
    path = Path(root).resolve(strict=True)
    if not path.is_dir() or (require_mount and not mount_checker(path)):
        raise ValueError("Admission root must be the verified mounted bucket")
    if any(path.iterdir()):
        raise ValueError("Admission bucket must be empty during initialization")
    metadata = _signed({"version": VERSION, "ledger_id": uuid.uuid4().hex,
                        "salt": secrets.token_hex(32)})
    _write_new_durable(path / "metadata.json", metadata, directory_sync)
    return metadata["ledger_id"]


initialize_bucket_store = initialize_bucket


@contextmanager
def _linux_flock(lock_path):
    try:
        import fcntl
    except ImportError:
        raise AdmissionDenied("The local admission lock is unavailable. Processing is paused.") from None
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        raise AdmissionDenied("The local admission lock failed. Processing is paused.") from None
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            raise AdmissionDenied("The demo is processing another request. Try again shortly.") from None
        except OSError:
            raise AdmissionDenied("The local admission lock failed. Processing is paused.") from None
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class MountedBucketAdmission:
    """Serialize one Space process and its same-host workers around durable slots."""

    def __init__(self, root, *, local_lock_path="/tmp/paper-evidence-admission.lock",
                 require_mount=True, mount_checker=os.path.ismount,
                 lock_provider=None, directory_sync=_sync_directory,
                 query_daily=8, visitor_query_daily=2,
                 upload_daily=100, visitor_upload_daily=2, cooldown=30,
                 clock=time.time):
        for daily, visitor in ((query_daily, visitor_query_daily), (upload_daily, visitor_upload_daily)):
            if type(daily) is not int or not 1 <= daily <= MAX_RECORDS:
                raise ValueError("Invalid daily admission limit")
            if type(visitor) is not int or not 1 <= visitor <= daily:
                raise ValueError("Invalid visitor admission limit")
        if type(cooldown) is not int or not 5 <= cooldown <= 3600:
            raise ValueError("Invalid admission cooldown")
        self.root = Path(root).resolve(strict=True)
        self.local_lock_path = Path(local_lock_path).resolve()
        self.require_mount = require_mount
        self.mount_checker = mount_checker
        self.lock_provider = lock_provider or _linux_flock
        self.directory_sync = directory_sync
        self.limits = {"query": (query_daily, visitor_query_daily),
                       "upload": (upload_daily, visitor_upload_daily)}
        self.cooldown, self.clock = cooldown, clock
        self._verify_mount()
        self.metadata = self._read_metadata()
        with _PROCESS_LOCKS_GUARD:
            self._process_lock = _PROCESS_LOCKS.setdefault(str(self.root), threading.Lock())
    def _verify_mount(self):
        if not self.root.is_dir() or (self.require_mount and not self.mount_checker(self.root)):
            raise AdmissionDenied("The private admission bucket is unavailable. Processing is paused.")

    def _verify_mount_each_call(self):
        try:
            self._verify_mount()
        except (OSError, ValueError):
            raise AdmissionDenied("The private admission bucket is unavailable. Processing is paused.") from None

    def _read_json(self, path):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise AdmissionDenied("The admission ledger is corrupt. Processing is paused.")
        try:
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("duplicate key")
                    result[key] = value
                return result
            value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            raise AdmissionDenied("The admission ledger is corrupt. Processing is paused.") from None
        if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
            raise AdmissionDenied("The admission ledger is corrupt. Processing is paused.")
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        if not hmac.compare_digest(value["sha256"], hashlib.sha256(_canonical(unsigned)).hexdigest()):
            raise AdmissionDenied("The admission ledger is corrupt. Processing is paused.")
        return value

    def _read_metadata(self):
        value = self._read_json(self.root / "metadata.json")
        if set(value) != {"version", "ledger_id", "salt", "sha256"}:
            raise AdmissionDenied("The admission ledger metadata is invalid. Processing is paused.")
        if (type(value["version"]) is not int or value["version"] != VERSION
                or not isinstance(value["ledger_id"], str)
                or not re.fullmatch(r"[0-9a-f]{32}", value["ledger_id"])
                or not isinstance(value["salt"], str) or not _HEX64.fullmatch(value["salt"])):
            raise AdmissionDenied("The admission ledger metadata is invalid. Processing is paused.")
        return value

    def _verify_metadata_unchanged(self):
        current = self._read_metadata()
        if (not hmac.compare_digest(current["ledger_id"], self.metadata["ledger_id"])
                or not hmac.compare_digest(current["salt"], self.metadata["salt"])):
            raise AdmissionDenied("The admission ledger identity changed. Processing is paused.")

    def _records(self, now):
        try:
            entries = list(self.root.iterdir())
        except OSError:
            raise AdmissionDenied("The private admission bucket cannot be read. Processing is paused.") from None
        if len(entries) - 1 > MAX_RECORDS:
            raise AdmissionDenied("The admission ledger reached its safe record limit. Processing is paused.")
        rows, expired, temporary = [], [], []
        for entry in entries:
            if entry.name == "metadata.json":
                continue
            if entry.name.startswith(".") and entry.name.endswith(".tmp") and _ID.fullmatch(entry.name[1:-4]):
                temporary.append(entry)
                continue
            match = _ID.fullmatch(entry.stem) if entry.suffix == ".json" else None
            if not match:
                raise AdmissionDenied("The admission ledger contains an unknown file. Processing is paused.")
            value = self._read_json(entry)
            legacy_fields = {"version", "id", "ledger_id", "salt_id", "visitor", "kind", "at", "sha256"}
            pending_fields = legacy_fields | {"state", "expires_at"}
            if set(value) not in (legacy_fields, pending_fields):
                raise AdmissionDenied("The admission ledger record is invalid. Processing is paused.")
            at = value["at"]
            if (type(value["version"]) is not int or value["version"] != VERSION or value["id"] != entry.stem
                    or value["ledger_id"] != self.metadata["ledger_id"]
                    or value["salt_id"] != hashlib.sha256(bytes.fromhex(self.metadata["salt"])).hexdigest()
                    or not _HEX64.fullmatch(value["visitor"] if isinstance(value["visitor"], str) else "")
                    or not isinstance(value["kind"], str) or value["kind"] not in self.limits
                    or type(at) not in (int, float)
                    or not math.isfinite(at) or at > now + 60):
                raise AdmissionDenied("The admission ledger record is invalid. Processing is paused.")
            state = value.get("state", "committed")
            if state not in ("pending", "committed"):
                raise AdmissionDenied("The admission ledger record is invalid. Processing is paused.")
            if state == "pending":
                expires_at = value["expires_at"]
                if type(expires_at) not in (int, float) or not math.isfinite(expires_at) or expires_at <= at:
                    raise AdmissionDenied("The admission ledger record is invalid. Processing is paused.")
                if expires_at <= now:
                    expired.append(entry)
                    continue
            if at <= now - 86400:
                expired.append(entry)
            else:
                rows.append(value)
        for entry in expired:
            try:
                entry.unlink()
            except OSError:
                raise AdmissionDenied("Expired admission records could not be retired safely.") from None
        for entry in temporary:
            try:
                entry.unlink()
            except OSError:
                raise AdmissionDenied("Incomplete admission updates could not be retired safely.") from None
        if expired or temporary:
            self._fsync_root()
        return rows

    def _fsync_root(self):
        try:
            self.directory_sync(self.root)
        except OSError:
            raise AdmissionDenied("The private bucket does not support durable directory sync. Processing is paused.") from None

    def status(self):
        """Return safe operational counts without exposing visitor identities."""
        self._verify_mount_each_call()
        if not self._process_lock.acquire(blocking=False):
            raise AdmissionDenied("The demo is processing another request. Try again shortly.")
        try:
            try:
                self.local_lock_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise AdmissionDenied("The local admission lock failed. Processing is paused.") from None
            with self.lock_provider(self.local_lock_path):
                self._verify_mount_each_call()
                self._verify_metadata_unchanged()
                now = self.clock()
                if type(now) not in (int, float) or not math.isfinite(now):
                    raise AdmissionDenied("The admission clock is unavailable. Processing is paused.")
                rows = self._records(now)
                return {"ledger_id": self.metadata["ledger_id"], "reservation_count": len(rows)}
        finally:
            self._process_lock.release()

    def usage(self, visitor):
        """Return rolling remaining counts; visitor identifiers stay private."""
        if not isinstance(visitor, str) or not visitor or len(visitor) > 1024:
            return {kind: {"session_remaining": None, "shared_remaining": None}
                    for kind in self.limits}
        self._verify_mount_each_call()
        if not self._process_lock.acquire(blocking=False):
            raise AdmissionDenied("Usage is temporarily unavailable while processing another request.")
        try:
            try:
                self.local_lock_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise AdmissionDenied("Usage is temporarily unavailable.") from None
            with self.lock_provider(self.local_lock_path):
                self._verify_metadata_unchanged()
                now = self.clock()
                if type(now) not in (int, float) or not math.isfinite(now):
                    raise AdmissionDenied("Usage is temporarily unavailable.")
                rows = self._records(now)
                visitor_hash = hmac.new(bytes.fromhex(self.metadata["salt"]),
                                        visitor.encode("utf-8"), hashlib.sha256).hexdigest()
                result = {}
                for kind, (shared_limit, session_limit) in self.limits.items():
                    recent = [row for row in rows if row["kind"] == kind]
                    own = sum(hmac.compare_digest(row["visitor"], visitor_hash) for row in recent)
                    result[kind] = {"session_remaining": max(0, session_limit - own),
                                    "shared_remaining": max(0, shared_limit - len(recent))}
                return result
        finally:
            self._process_lock.release()

    @contextmanager
    def admit(self, visitor, kind="query"):
        if kind not in self.limits:
            raise ValueError("Unknown admission operation")
        if not isinstance(visitor, str) or not visitor or len(visitor) > 1024:
            raise AdmissionDenied("Cannot identify this visitor. Saved examples and browsing remain available.")
        self._verify_mount_each_call()
        if not self._process_lock.acquire(blocking=False):
            raise AdmissionDenied("The demo is processing another request. Try again shortly.")
        try:
            try:
                self.local_lock_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise AdmissionDenied("The local admission lock failed. Processing is paused.") from None
            with self.lock_provider(self.local_lock_path):
                self._verify_mount_each_call()
                self._verify_metadata_unchanged()
                now = self.clock()
                if type(now) not in (int, float) or not math.isfinite(now):
                    raise AdmissionDenied("The admission clock is unavailable. Processing is paused.")
                rows = self._records(now)
                daily, visitor_daily = self.limits[kind]
                visitor_hash = hmac.new(bytes.fromhex(self.metadata["salt"]),
                                        visitor.encode("utf-8"), hashlib.sha256).hexdigest()
                recent = [row for row in rows if row["kind"] == kind]
                own = [row for row in recent if hmac.compare_digest(row["visitor"], visitor_hash)]
                if len(recent) >= daily:
                    raise AdmissionDenied("The demo has reached its current processing capacity. Try again later; saved examples and browsing remain available.")
                if len(own) >= visitor_daily:
                    operation = "question" if kind == "query" else "PDF processing"
                    raise AdmissionDenied(f"No {operation} allowance remains right now. Saved examples and browsing remain available.")
                if own and now - max(row["at"] for row in own) < self.cooldown:
                    raise AdmissionDenied("Please wait before submitting again. Saved examples and browsing remain available.")
                if len(rows) >= MAX_RECORDS:
                    raise AdmissionDenied("The admission ledger reached its safe record limit. Processing is paused.")
                record_id = secrets.token_hex(16)
                record = _signed({"version": VERSION, "id": record_id,
                    "ledger_id": self.metadata["ledger_id"],
                    "salt_id": hashlib.sha256(bytes.fromhex(self.metadata["salt"])).hexdigest(),
                    "visitor": visitor_hash, "kind": kind, "at": now,
                    "state": "pending", "expires_at": now + RESERVATION_TTL})
                try:
                    _write_new_durable(self.root / (record_id + ".json"), record,
                                       self.directory_sync)
                except (OSError, ValueError):
                    raise AdmissionDenied("The admission reservation could not be stored durably. Processing is paused.") from None
                reservation = _Reservation(self, self.root / (record_id + ".json"), record)
                try:
                    yield reservation
                except BaseException:
                    reservation.rollback()
                    raise
                else:
                    if not reservation.committed:
                        reservation.rollback()
        finally:
            self._process_lock.release()


class _Reservation:
    def __init__(self, admission, path, record):
        self.admission, self.path, self.record = admission, path, record
        self.committed = False

    def commit(self):
        if self.committed:
            return
        unsigned = {key: value for key, value in self.record.items() if key not in ("sha256", "state", "expires_at")}
        temporary = self.path.with_name("." + self.path.stem + ".tmp")
        with temporary.open("xb") as handle:
            handle.write(_canonical(_signed(unsigned)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        self.admission._fsync_root()
        self.committed = True

    def rollback(self):
        if self.committed:
            return
        try:
            self.path.unlink(missing_ok=True)
            self.admission._fsync_root()
        except (OSError, AdmissionDenied):
            # A surviving pending record remains bounded by its expiry.
            pass
