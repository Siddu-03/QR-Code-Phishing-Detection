"""
system_info.py
===============
Evaluation Framework — environment/hardware information collection.

Collects a snapshot of the machine an evaluation run executed on (OS, CPU,
RAM, Python/OpenCV/NumPy versions, worker count) so results can be traced
back to the hardware/software context that produced them. This is written
into ``benchmark.json`` and surfaced in both reports.

Every field degrades gracefully to ``None``/``"unknown"`` rather than
raising if a given piece of information isn't available in the current
environment (e.g. no ``/proc/meminfo`` on some platforms).
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class SystemInfo:
    """A snapshot of the environment an evaluation run executed on."""

    os_name: str
    os_release: str
    machine: str
    python_version: str
    cpu_count: int | None
    total_ram_mb: float | None
    opencv_version: str | None
    numpy_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pretty_print(self) -> str:
        ram = f"{self.total_ram_mb:.0f} MB" if self.total_ram_mb is not None else "unknown"
        lines = [
            "=" * 60,
            "  System information",
            "=" * 60,
            f"  OS             : {self.os_name} {self.os_release}",
            f"  Machine        : {self.machine}",
            f"  Python         : {self.python_version}",
            f"  CPU cores      : {self.cpu_count if self.cpu_count is not None else 'unknown'}",
            f"  RAM            : {ram}",
            f"  OpenCV         : {self.opencv_version or 'not installed'}",
            f"  NumPy          : {self.numpy_version or 'not installed'}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _total_ram_mb() -> float | None:
    """Best-effort total system RAM in MB, without adding a psutil dependency.

    Tries ``os.sysconf`` (POSIX) first, then ``/proc/meminfo`` (Linux) as a
    fallback; returns ``None`` if neither is available (e.g. Windows
    without ``psutil`` installed).
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return (page_size * page_count) / (1024 * 1024)
    except (ValueError, AttributeError, OSError):
        pass

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass

    return None


def _opencv_version() -> str | None:
    try:
        import cv2

        return cv2.__version__
    except ImportError:
        return None


def _numpy_version() -> str | None:
    try:
        import numpy

        return numpy.__version__
    except ImportError:
        return None


def collect_system_info() -> SystemInfo:
    """Collect and return a :class:`SystemInfo` snapshot for the current machine.

    Never raises: every field falls back to ``None`` if its detection
    method fails, so this can always be safely included in a run's outputs.
    """
    return SystemInfo(
        os_name=platform.system() or "unknown",
        os_release=platform.release() or "unknown",
        machine=platform.machine() or "unknown",
        python_version=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        total_ram_mb=_total_ram_mb(),
        opencv_version=_opencv_version(),
        numpy_version=_numpy_version(),
    )