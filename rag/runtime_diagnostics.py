
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


DIAGNOSTICS_ENABLED = (
    os.getenv("RUNTIME_DIAGNOSTICS", "false").lower() == "true"
)


def get_rss_mb() -> float | None:
    """
    Return current process RSS in MiB on Linux.

    On unsupported platforms return None.
    Does not require additional dependencies.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")

        with open("/proc/self/statm", "r", encoding="utf-8") as file:
            fields = file.read().split()

        resident_pages = int(fields[1])

        return resident_pages * page_size / (1024 ** 2)

    except (OSError, AttributeError, ValueError, IndexError):
        return None


def format_memory(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2f} MiB"


def log_event(message: str) -> None:
    """
    Print a diagnostic message to runtime logs.
    """
    if not DIAGNOSTICS_ENABLED:
        return

    timestamp = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )

    print(
        f"[RUNTIME] [{timestamp}] {message}",
        flush=True,
    )


@contextmanager
def measure_stage(
    stage_name: str,
) -> Iterator[None]:
    """
    Measure execution time, CPU time and process memory
    for an individual pipeline stage.
    """
    if not DIAGNOSTICS_ENABLED:
        yield
        return

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rss_start = get_rss_mb()

    status = "OK"

    log_event(
        f"START | {stage_name} | "
        f"rss={format_memory(rss_start)}"
    )

    try:
        yield

    except BaseException as exc:
        status = "ERROR"

        log_event(
            f"ERROR | {stage_name} | "
            f"exception={type(exc).__name__}"
        )

        raise

    finally:
        wall_time = time.perf_counter() - wall_start
        cpu_time = time.process_time() - cpu_start
        rss_end = get_rss_mb()

        cpu_percent = (
            cpu_time / wall_time * 100
            if wall_time > 0
            else 0.0
        )

        rss_delta = (
            rss_end - rss_start
            if rss_start is not None and rss_end is not None
            else None
        )

        log_event(
            f"END | {stage_name} | "
            f"status={status} | "
            f"wall_time={wall_time:.3f}s | "
            f"cpu_time={cpu_time:.3f}s | "
            f"cpu_avg={cpu_percent:.1f}% | "
            f"rss_start={format_memory(rss_start)} | "
            f"rss_end={format_memory(rss_end)} | "
            f"rss_delta={format_memory(rss_delta)}"
        )
