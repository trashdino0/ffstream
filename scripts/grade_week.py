"""Compatibility wrapper for the old script path.

The grader now lives in ``ffstream.grade`` so the CLI and the weekly runner can
share it. Existing commands keep working:

    python scripts/grade_week.py --season 2026 --week 10
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ffstream.grade import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
