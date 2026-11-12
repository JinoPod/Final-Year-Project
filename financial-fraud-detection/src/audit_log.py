"""
Logging & Audit Services layer (report Section 3.6.5).

Every scored transaction is written to an SQLite database with:
  - the decision log: input values, fraud probability, threshold, decision,
    model name and version;
  - the audit trail: the SHAP explanation of that decision (per-feature
    contributions and base value), so an auditor can review the reasoning
    later without re-running the model;
  - a SHA-256 hash of the input values, so a reviewer can check that the
    logged input is the one that was scored.

The log is append-only: database triggers reject UPDATE and DELETE on the
decision and batch tables. Nothing is ever deleted: start_fresh_log() moves the
whole database file into logs/archive/ (kept intact) and a new, empty log begins.
"""

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.inference import top_factors

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    source_file         TEXT,
    file_sha256         TEXT,
    model_name          TEXT    NOT NULL,
    model_version       TEXT    NOT NULL,
    threshold           REAL    NOT NULL,
    feature_names_json  TEXT    NOT NULL,
    total_rows          INTEGER NOT NULL,
    scored_rows         INTEGER NOT NULL,
    rejected_rows       INTEGER NOT NULL,
    flagged_rows        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id            INTEGER NOT NULL REFERENCES batches(id),
    row_number          INTEGER NOT NULL,
    fraud_probability   REAL    NOT NULL,
    flagged             INTEGER NOT NULL,
    actual              INTEGER,
    top_factors         TEXT    NOT NULL,
    features_json       TEXT    NOT NULL,
    shap_json           TEXT    NOT NULL,
    base_value          REAL    NOT NULL,
    input_sha256        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_batch ON decisions(batch_id);
CREATE INDEX IF NOT EXISTS idx_decisions_flagged ON decisions(flagged);
"""

APPEND_ONLY_TRIGGERS = [
    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
    f"BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;"
    for table in ("batches", "decisions") for op in ("UPDATE", "DELETE")
]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_values(values):
    """Stable SHA-256 of a row of feature values."""
    payload = json.dumps([float(v) for v in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLog:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            for trigger in APPEND_ONLY_TRIGGERS:
                conn.execute(trigger)

    @contextmanager
    def _connect(self):
        """Commit on success, roll back on error, and always close the file handle
        (Windows cannot move a database file that still has an open connection)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---- writing --------------------------------------------------------

    def log_batch(self, batch, source_file=None, file_bytes=None, total_rows=None, rejected_rows=0):
        """Write a ScoredBatch (from src.inference) to the log. Returns the batch id."""
        feature_names = list(batch.raw_features.columns)
        raw = batch.raw_features.to_numpy(dtype=float)
        res = batch.results
        has_actual = "actual" in res.columns

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO batches (created_at, source_file, file_sha256, model_name,
                       model_version, threshold, feature_names_json, total_rows,
                       scored_rows, rejected_rows, flagged_rows)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (_now(), source_file,
                 hashlib.sha256(file_bytes).hexdigest() if file_bytes is not None else None,
                 batch.model_name, batch.model_version, batch.threshold,
                 json.dumps(feature_names),
                 int(total_rows if total_rows is not None else len(res)),
                 int(len(res)), int(rejected_rows), int(res["flagged"].sum())),
            )
            batch_id = cur.lastrowid
            conn.executemany(
                """INSERT INTO decisions (batch_id, row_number, fraud_probability, flagged,
                       actual, top_factors, features_json, shap_json, base_value, input_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (batch_id, int(res["row"].iat[i]), float(res["fraud_probability"].iat[i]),
                     int(res["flagged"].iat[i]),
                     int(res["actual"].iat[i]) if has_actual else None,
                     top_factors(batch.shap_values[i], raw[i], feature_names),
                     json.dumps(raw[i].tolist()),
                     json.dumps(np.round(batch.shap_values[i], 6).tolist()),
                     batch.base_value, hash_values(raw[i]))
                    for i in range(len(res))
                ],
            )
        return batch_id

    # ---- reading --------------------------------------------------------

    def list_batches(self):
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM batches ORDER BY id DESC", conn)

    def batch_summaries(self, review_floor):
        """One row per batch (newest first) with its counts per risk level."""
        query = """SELECT b.id, b.created_at, b.source_file, b.file_sha256, b.threshold,
                          b.total_rows, b.scored_rows, b.rejected_rows,
                          COALESCE(SUM(d.fraud_probability >= b.threshold), 0) AS high,
                          COALESCE(SUM(d.fraud_probability >= ? AND d.fraud_probability < b.threshold), 0)
                              AS review
                   FROM batches b LEFT JOIN decisions d ON d.batch_id = b.id
                   GROUP BY b.id ORDER BY b.id DESC"""
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=[float(review_floor)])

    def find_file(self, file_bytes):
        """Earlier batches made from exactly this file (matched by SHA-256), newest first."""
        digest = hashlib.sha256(file_bytes).hexdigest()
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT id, created_at, source_file FROM batches WHERE file_sha256 = ? ORDER BY id DESC",
                conn, params=[digest])

    def get_decisions(self, batch_id=None, flagged_only=False, limit=None):
        """Decision log as a DataFrame (without the bulky JSON columns)."""
        query = """SELECT d.id, d.batch_id, b.created_at, b.source_file, d.row_number,
                          b.model_name, b.model_version, b.threshold, d.fraud_probability,
                          d.flagged, d.actual, d.top_factors, d.input_sha256
                   FROM decisions d JOIN batches b ON b.id = d.batch_id WHERE 1=1"""
        params = []
        if batch_id is not None:
            query += " AND d.batch_id = ?"
            params.append(int(batch_id))
        if flagged_only:
            query += " AND d.flagged = 1"
        query += " ORDER BY d.fraud_probability DESC"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        df["flagged"] = df["flagged"].astype(bool)
        return df

    def get_decision(self, decision_id):
        """One decision with its stored explanation, ready to plot."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT d.*, b.created_at, b.source_file, b.model_name, b.model_version,
                          b.threshold, b.feature_names_json
                   FROM decisions d JOIN batches b ON b.id = d.batch_id WHERE d.id = ?""",
                (int(decision_id),),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["features"] = np.array(json.loads(d.pop("features_json")))
        d["shap_values"] = np.array(json.loads(d.pop("shap_json")))
        d["feature_names"] = json.loads(d.pop("feature_names_json"))
        d["flagged"] = bool(d["flagged"])
        d["hash_ok"] = hash_values(d["features"]) == d["input_sha256"]
        return d


def start_fresh_log(db_path, archive_dir=None):
    """Move the current log into an archive folder and leave no file behind, so the next
    AuditLog(db_path) starts empty. Nothing is deleted. Returns the archive path, or None
    if there was no log to archive."""
    if not os.path.exists(db_path):
        return None
    archive_dir = archive_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)), "archive")
    os.makedirs(archive_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    stem, ext = os.path.splitext(os.path.basename(db_path))
    target = os.path.join(archive_dir, f"{stem}_{stamp}{ext}")
    shutil.move(db_path, target)
    return target


def list_archives(db_path, archive_dir=None):
    """Archived log files, newest first."""
    archive_dir = archive_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)), "archive")
    if not os.path.isdir(archive_dir):
        return []
    return sorted((f for f in os.listdir(archive_dir) if f.endswith(".db")), reverse=True)
