"""Server-side admission shared by UI and API, before expensive operations.

SQLite requires a single host and a filesystem with working transactions/locks.
An object-storage mount is not sufficient evidence of those guarantees.
"""
from contextlib import closing, contextmanager
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
import time


class AdmissionDenied(RuntimeError):
    pass


class _Reservation:
    def __init__(self, admission, reservation):
        self.admission, self.reservation, self.committed = admission, reservation, False

    def commit(self):
        with self.admission._connect() as db:
            updated = db.execute("UPDATE admissions SET active_until=0 WHERE id=?", (self.reservation,))
            if updated.rowcount != 1:
                raise sqlite3.IntegrityError("The pending usage reservation is missing")
        self.committed = True

    def rollback(self):
        if not self.committed:
            try:
                with self.admission._connect() as db:
                    db.execute("DELETE FROM admissions WHERE id=?", (self.reservation,))
            except (sqlite3.Error, OSError):
                # Pending leases expire after one hour if durable cleanup fails.
                pass


class Admission:
    def __init__(self, path, *, initialize=False, daily=8, visitor_daily=2,
                 upload_daily=100, upload_visitor_daily=2, cooldown=30, clock=time.time):
        if type(daily) is not int or not 1 <= daily <= 100:
            raise ValueError("Invalid global daily limit")
        if type(visitor_daily) is not int or not 1 <= visitor_daily <= daily:
            raise ValueError("Invalid visitor daily limit")
        if type(upload_daily) is not int or not 1 <= upload_daily <= 100:
            raise ValueError("Invalid global PDF processing limit")
        if type(upload_visitor_daily) is not int or not 1 <= upload_visitor_daily <= upload_daily:
            raise ValueError("Invalid visitor PDF processing limit")
        if not 5 <= cooldown <= 3600:
            raise ValueError("Invalid admission cooldown")
        self.path = Path(path)
        self.daily, self.visitor_daily = daily, visitor_daily
        self.limits = {"query": (daily, visitor_daily),
                       "upload": (upload_daily, upload_visitor_daily)}
        self.cooldown, self.clock = cooldown, clock
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("INSERT OR IGNORE INTO metadata VALUES ('salt', ?)", (secrets.token_hex(32),))
                db.execute("CREATE TABLE IF NOT EXISTS admissions (id TEXT PRIMARY KEY, visitor TEXT NOT NULL, kind TEXT NOT NULL, at REAL NOT NULL, active_until REAL NOT NULL)")
        # mode=rw prevents a lost volume or removed database from silently resetting
        # the allowance. Startup and every reservation fail closed instead.
        with self._connect() as db:
            salt = db.execute("SELECT value FROM metadata WHERE key='salt'").fetchone()
            if not salt or len(salt[0]) != 64:
                raise ValueError("Admission database is not initialized")
            self.salt = bytes.fromhex(salt[0])

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=2)) as db, db:
            yield db

    def usage(self, visitor):
        """Return remaining allowances without exposing the hashed visitor key."""
        if not isinstance(visitor, str) or not visitor or len(visitor) > 1024:
            return {kind: {"session_remaining": None, "shared_remaining": None}
                    for kind in ("query", "upload")}
        identity = hmac.new(self.salt, visitor.encode(), hashlib.sha256).hexdigest()
        now = self.clock()
        result = {}
        with self._connect() as db:
            for kind in ("query", "upload"):
                total, own = self.limits[kind]
                shared_used = db.execute("SELECT COUNT(*) FROM admissions WHERE kind=? AND at>? AND (active_until=0 OR at>?)", (kind, now - 86400, now - 3600)).fetchone()[0]
                own_used = db.execute("SELECT COUNT(*) FROM admissions WHERE visitor=? AND kind=? AND at>? AND (active_until=0 OR at>?)", (identity, kind, now - 86400, now - 3600)).fetchone()[0]
                result[kind] = {"session_remaining": max(0, own - own_used),
                                "shared_remaining": max(0, total - shared_used)}
        return result

    @contextmanager
    def admit(self, visitor, kind="query"):
        if kind not in {"query", "upload"}:
            raise ValueError("Unknown admission operation")
        if not isinstance(visitor, str) or not visitor or len(visitor) > 1024:
            raise AdmissionDenied("Cannot identify this visitor. Reload the app. Saved examples and browsing remain available.")
        identity = hmac.new(self.salt, visitor.encode(), hashlib.sha256).hexdigest()
        reservation = secrets.token_hex(16)
        now = self.clock()
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                # Committed history expires after 48h for ledger hygiene. A
                # pending crash reservation has a bounded one-hour lease.
                db.execute("DELETE FROM admissions WHERE (at < ? AND active_until=0) OR (at < ? AND active_until!=0)", (now - 172800, now - 3600))
                # Rolling 24-hour windows avoid a burst across a UTC midnight.
                total = db.execute("SELECT COUNT(*) FROM admissions WHERE kind=? AND at>?", (kind, now - 86400)).fetchone()[0]
                own = db.execute("SELECT COUNT(*), MAX(at) FROM admissions WHERE visitor=? AND kind=? AND at>?", (identity, kind, now - 86400)).fetchone()
                daily, visitor_daily = self.limits[kind]
                if total >= daily:
                    raise AdmissionDenied("The demo has reached its current processing capacity. Try again later; saved examples and browsing remain available.")
                if own[0] >= visitor_daily:
                    operation = "question" if kind == "query" else "PDF processing"
                    raise AdmissionDenied(f"No {operation} allowance remains right now. Saved examples and browsing remain available.")
                if own[1] is not None and now - own[1] < self.cooldown:
                    raise AdmissionDenied("Please wait before submitting again. Saved examples and browsing remain available.")
                if db.execute("SELECT COUNT(*) FROM admissions WHERE active_until!=0").fetchone()[0]:
                    raise AdmissionDenied("The demo is processing another request. Try again shortly.")
                db.execute("INSERT INTO admissions VALUES (?, ?, ?, ?, ?)", (reservation, identity, kind, now, 1))
        except AdmissionDenied:
            raise
        except (sqlite3.Error, OSError):
            raise AdmissionDenied("The usage limiter is unavailable. New processing is paused; saved examples and browsing remain available.") from None
        reservation_context = _Reservation(self, reservation)
        try:
            yield reservation_context
        except BaseException:
            reservation_context.rollback()
            raise
        else:
            if not reservation_context.committed:
                reservation_context.rollback()
