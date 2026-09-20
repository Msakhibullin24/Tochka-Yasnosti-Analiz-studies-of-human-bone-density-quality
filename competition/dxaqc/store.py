"""Local durable jobs and append-only expert reviews (schema version 1)."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "jobs.sqlite3"
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError(f"unsupported database schema {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, input TEXT NOT NULL,
                    out TEXT NOT NULL, upload_dir TEXT, done INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0, summary TEXT, error TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS reviews (
                    job_id TEXT NOT NULL REFERENCES jobs(id), row_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, document TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(job_id, row_id, revision));
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        result["summary"] = json.loads(result["summary"]) if result["summary"] else None
        return result

    def add(self, job_id, input_path, out, upload_dir=None):
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,status,input,out,upload_dir,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                       (job_id, "queued", str(input_path), str(out), str(upload_dir) if upload_dir else None, now, now))
        return self.get(job_id)

    def get(self, job_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit=50, offset=0):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id LIMIT ? OFFSET ?", (limit, offset))]

    def update(self, job_id, **fields):
        allowed = {"status", "done", "total", "summary", "error", "cancel_requested"}
        if not fields or not fields.keys() <= allowed:
            raise ValueError("invalid job fields")
        if "summary" in fields:
            fields["summary"] = json.dumps(fields["summary"], ensure_ascii=False)
        fields["updated_at"] = time.time()
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?",
                       (*fields.values(), job_id))

    def cancel(self, job_id):
        with self.connect() as db:
            return db.execute("UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=? AND status IN ('queued','running')",
                              (time.time(), job_id)).rowcount > 0

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (time.time(), row["id"]))
            result = self.decode(row)
            result["status"] = "running"
            return result

    def interrupted(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs WHERE status='running'")]

    def reviews(self, job_id, row_id):
        with self.connect() as db:
            return [{"revision": r["revision"], "created_at": r["created_at"], **json.loads(r["document"])}
                    for r in db.execute("SELECT * FROM reviews WHERE job_id=? AND row_id=? ORDER BY revision",
                                        (job_id, row_id))]

    def latest_reviews(self, job_id):
        """Read queue state in one snapshot, including drafts after confirmations."""
        with self.connect() as db:
            return {r["row_id"]: {"revision": r["revision"], **json.loads(r["document"])}
                    for r in db.execute("""
                        SELECT r.* FROM reviews r
                        JOIN (SELECT row_id, MAX(revision) AS revision FROM reviews
                              WHERE job_id=? GROUP BY row_id) latest
                        ON r.row_id=latest.row_id AND r.revision=latest.revision
                        WHERE r.job_id=?
                    """, (job_id, job_id))}

    def review(self, job_id, row_id, expected_revision, document):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute("SELECT COALESCE(MAX(revision),0) FROM reviews WHERE job_id=? AND row_id=?",
                                  (job_id, row_id)).fetchone()[0]
            if revision != expected_revision:
                raise Conflict("review was changed; reload the latest revision")
            db.execute("INSERT INTO reviews VALUES(?,?,?,?,?)", (job_id, row_id, revision + 1,
                       json.dumps(document, ensure_ascii=False, allow_nan=False), time.time()))
        return self.reviews(job_id, row_id)[-1]
