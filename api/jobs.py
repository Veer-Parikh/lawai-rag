"""
api/jobs.py — Minimal in-process background jobs with progress, one running job per kind.

Used for admin actions that take tens of seconds (catalog builds, source
checks). Jobs live in memory; their durable results are written by the work
itself (catalog, change log).
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

ProgressFn = Callable[[int, int, str], None]


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "running"           # running | completed | failed
    requested_by: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    progress_done: int = 0
    progress_total: int = 0
    current: str = ""
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, work: Callable[[ProgressFn], Any], *, requested_by: str) -> Job:
        with self._lock:
            if any(job.kind == kind and job.status == "running" for job in self._jobs.values()):
                raise RuntimeError(f"A {kind.replace('_', ' ')} job is already running")
            job = Job(job_id=str(uuid.uuid4()), kind=kind, requested_by=requested_by)
            self._jobs[job.job_id] = job

        def progress(done: int, total: int, current: str) -> None:
            job.progress_done, job.progress_total, job.current = done, total, current

        def runner() -> None:
            try:
                job.result = work(progress)
                job.status = "completed"
            except Exception as exc:  # surfaced to the admin UI, never swallowed
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()
                job.current = ""

        threading.Thread(target=runner, daemon=True).start()
        return job

    def latest(self, kind: str) -> Job | None:
        jobs = [job for job in self._jobs.values() if job.kind == kind]
        return max(jobs, key=lambda job: job.started_at) if jobs else None

    def get(self, job_id: str) -> Job:
        return self._jobs[job_id]


jobs = JobRegistry()
