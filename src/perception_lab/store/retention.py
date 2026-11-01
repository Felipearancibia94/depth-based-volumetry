"""Retention: delete old evidence images and old event rows.

Media folders are named by day, so expiry is decided by folder name; event rows by their
creation time. Run regularly (``lab purge``); ``dry_run`` reports without deleting.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from perception_lab.store.db import EventStore


@dataclass
class PurgeReport:
    media_days_removed: list[str] = field(default_factory=list)
    events_removed: int = 0
    dry_run: bool = False


def purge(
    store: EventStore,
    media_dir: str | Path,
    *,
    media_retention_days: float,
    event_retention_days: float,
    now_unix: float | None = None,
    dry_run: bool = False,
) -> PurgeReport:
    now = time.time() if now_unix is None else now_unix
    report = PurgeReport(dry_run=dry_run)
    media_cutoff = time.strftime("%Y-%m-%d", time.localtime(now - media_retention_days * 86400.0))
    root = Path(media_dir)
    if root.is_dir():
        for day in sorted(p for p in root.iterdir() if p.is_dir()):
            if _is_day(day.name) and day.name < media_cutoff:
                report.media_days_removed.append(day.name)
                if not dry_run:
                    shutil.rmtree(day)
    event_cutoff = now - event_retention_days * 86400.0
    if dry_run:
        report.events_removed = len(store.list(limit=1_000_000, created_before_unix=event_cutoff))
    else:
        report.events_removed = store.delete_created_before(event_cutoff)
    return report


def _is_day(name: str) -> bool:
    try:
        time.strptime(name, "%Y-%m-%d")
    except ValueError:
        return False
    return True
