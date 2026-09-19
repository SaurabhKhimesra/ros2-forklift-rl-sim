"""Run logging: console, CSV and TensorBoard from one call site.

The previous logger called ``logging.getLogger('MyCustomLogger')`` regardless of
the ``name`` argument, so every instance shared one underlying logger and each
construction stacked another handler onto it -- duplicate lines, growing without
bound. It also emitted a line per tensorboard key on *every* environment step,
roughly ten synchronous file writes per step.

This one batches scalars, writes a machine-readable ``metrics.csv`` alongside the
TensorBoard event file, and prints a compact aligned table at a configurable
interval.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_TB_AVAILABLE = True
try:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional
    _TB_AVAILABLE = False
    SummaryWriter = None  # type: ignore[assignment]


class RunLogger:
    """Writes scalars to TensorBoard + CSV and prints periodic summaries."""

    def __init__(
        self, run_dir: str | Path, use_tensorboard: bool = True, quiet: bool = False
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self._start = time.time()

        self._writer = None
        if use_tensorboard and _TB_AVAILABLE:
            self._writer = SummaryWriter(str(self.run_dir / "tb"))

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv: csv.DictWriter | None = None
        self._columns: list[str] = []

    # ------------------------------------------------------------------ #
    def log(self, step: int, metrics: Mapping[str, float], print_it: bool = False) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if self._writer is not None:
            for key, value in clean.items():
                self._writer.add_scalar(key, value, step)

        row = {"step": step, "elapsed_s": round(time.time() - self._start, 2), **clean}
        if self._csv is None:
            self._columns = list(row)
            self._csv = csv.DictWriter(self._csv_file, fieldnames=self._columns)
            self._csv.writeheader()
        elif set(row) - set(self._columns):
            # New keys appear when evaluation first runs; rewrite the header
            # rather than silently dropping the columns.
            self._columns += [k for k in row if k not in self._columns]
            self._rewrite_with_columns()
        self._csv.writerow({k: row.get(k, "") for k in self._columns})
        self._csv_file.flush()

        if print_it and not self.quiet:
            self._print(step, clean)

    def _rewrite_with_columns(self) -> None:
        existing = self._csv_path.read_text().splitlines()
        body = existing[1:] if existing else []
        old_cols = existing[0].split(",") if existing else []
        self._csv_file.close()
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self._columns)
        self._csv.writeheader()
        for line in body:
            values = next(csv.reader([line]))
            self._csv.writerow(dict(zip(old_cols, values, strict=False)))

    def _print(self, step: int, metrics: Mapping[str, float]) -> None:
        elapsed = time.time() - self._start
        head = f"step {step:>9,}  {elapsed / 60:6.1f} min"
        body = "  ".join(f"{k.split('/')[-1]}={v:+.3f}" for k, v in metrics.items())
        print(f"{head} | {body}", file=sys.stdout, flush=True)

    def log_text(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        if not self._csv_file.closed:
            self._csv_file.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
