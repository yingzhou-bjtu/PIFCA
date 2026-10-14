#!/usr/bin/env python3
"""Validate the public PIFCA repository layout."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_MEDIA_SUFFIXES = {".ipynb", ".pdf", ".png", ".jpg", ".jpeg"}


def main() -> int:
    expected_dirs = {
        "assets",
        "assets/figures",
        "assets/reports",
        "dataset",
        "docs",
        "notebooks",
        "system",
    }
    missing_dirs = sorted(name for name in expected_dirs if not (ROOT / name).is_dir())
    root_artifacts = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in ROOT_MEDIA_SUFFIXES
    )
    notebook_count = len(list((ROOT / "notebooks").glob("*.ipynb")))
    figure_count = len(
        [
            path
            for path in (ROOT / "assets" / "figures").iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    report_count = len(
        [
            path
            for path in (ROOT / "assets" / "reports").iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
    )
    report = {
        "missing_dirs": missing_dirs,
        "root_artifacts": root_artifacts,
        "notebook_count": notebook_count,
        "figure_count": figure_count,
        "report_count": report_count,
    }
    print(json.dumps(report, indent=2))

    if missing_dirs or root_artifacts or notebook_count != 5 or figure_count < 10 or report_count < 5:
        return 1
    print("PIFCA_LAYOUT_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
