"""CLI entrypoint for building and executing the diffgeom research notebooks."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def build_notebooks_main() -> None:
    """Build and execute all (or selected) differential geometry notebooks."""
    parser = argparse.ArgumentParser(
        prog="diffgeom-build-notebooks",
        description="Build and execute the 9 differential-geometry notebooks.",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Optional subset of notebook numbers (1..9) to rebuild; defaults to all 9.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "build_all_notebooks.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not locate {script_path}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    mod = runpy.run_path(str(script_path), run_name="diffgeom._notebook_builder")
    targets = args.notebooks or [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for idx in targets:
        mod[f"build_notebook_{idx:02d}"]()
