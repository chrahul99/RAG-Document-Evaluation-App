import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import AppConfig


def initialize_history_db(config: AppConfig) -> None:
    """Create the SQLite table used for Q&A and metric history."""
    Path(config.history_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.history_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                response_time_seconds REAL NOT NULL,
                retrieval_latency_seconds REAL NOT NULL
            )
            """
        )


def save_qa_record(
    config: AppConfig,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    metrics: dict[str, Any],
    response_time_seconds: float,
    retrieval_latency_seconds: float,
) -> None:
    initialize_history_db(config)
    with sqlite3.connect(config.history_db_path) as connection:
        connection.execute(
            """
            INSERT INTO qa_history (
                created_at,
                question,
                answer,
                sources_json,
                metrics_json,
                response_time_seconds,
                retrieval_latency_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                answer,
                json.dumps(sources),
                json.dumps(metrics),
                response_time_seconds,
                retrieval_latency_seconds,
            ),
        )


def load_qa_history(config: AppConfig, limit: int = 100) -> list[dict[str, Any]]:
    initialize_history_db(config)
    with sqlite3.connect(config.history_db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *
            FROM qa_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = []
    for row in rows:
        record = dict(row)
        record["sources"] = json.loads(record.pop("sources_json"))
        record["metrics"] = json.loads(record.pop("metrics_json"))
        records.append(record)
    return records
