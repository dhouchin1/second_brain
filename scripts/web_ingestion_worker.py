#!/usr/bin/env python3
"""Background worker that processes queued web ingestion jobs."""

import asyncio
import logging
import os
import sqlite3

from config import settings
from services.web_ingestion_service import WebIngestionService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def get_conn():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


async def run_worker():
    service = WebIngestionService(get_conn)
    if not service.job_queue or not service.job_queue._client:
        logging.error("Redis is not configured; cannot start web ingestion worker.")
        return

    logging.info("Web ingestion worker started (queue: web_ingestion:jobs)")

    while True:
        payload = await service.job_queue.dequeue("web_ingestion:jobs", block=True, timeout=5)
        if not payload:
            continue
        job_id = payload.get("job_id")
        try:
            await service.process_job_payload(payload)
            logging.info("Processed ingestion job %s", job_id)
        except Exception as exc:  # pragma: no cover
            logging.exception("Ingestion job %s failed: %s", job_id, exc)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_worker())
