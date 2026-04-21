"""Pin large on-disk caches to a project disk (not the small overlay root)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Large shared disk; keeps HF / tmp / openpi / wandb / torch caches off overlay.
TGY_DISK_ROOT = Path("/mnt/project_rlinf/tgy")


def apply_tgy_disk_caches(*, log: bool = True) -> dict[str, str]:
    """Set cache-related env vars under ``TGY_DISK_ROOT`` and optionally log them.

    Called at PyTorch training entry startup so HuggingFace, temp files, and
    related tooling do not fill the container overlay (``/``).
    """
    root = TGY_DISK_ROOT
    layout: list[tuple[str, Path]] = [
        ("HF_HOME", root / ".cache" / "huggingface"),
        ("HF_DATASETS_CACHE", root / ".cache" / "huggingface" / "datasets"),
        ("HF_HUB_CACHE", root / ".cache" / "huggingface" / "hub"),
        ("OPENPI_DATA_HOME", root / ".cache" / "openpi"),
        ("TMPDIR", root / "tmp"),
        ("WANDB_DIR", root / ".wandb"),
        ("TORCH_HOME", root / ".cache" / "torch"),
    ]
    resolved: dict[str, str] = {}
    for key, path in layout:
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        resolved[key] = str(path)

    if log:
        log_lines = [f"Disk cache layout (root={root}):"]
        for key, val in resolved.items():
            log_lines.append(f"  {key}={val}")
        logging.info("\n".join(log_lines))

    return resolved
