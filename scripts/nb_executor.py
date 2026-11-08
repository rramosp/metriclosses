"""Helper to construct and execute IPython notebooks (.ipynb) with inline outputs and plots."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys
import textwrap
import traceback
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_markdown_cell(source: str) -> Dict[str, Any]:
    dedented = textwrap.dedent(source).strip()
    lines = [line + "\n" for line in dedented.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def make_code_cell(
    source: str,
    execution_count: int,
    outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    dedented = textwrap.dedent(source).strip()
    lines = [line + "\n" for line in dedented.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": lines,
    }


def build_and_execute_notebook(
    output_path: str | Path,
    cells_spec: List[Dict[str, str]],
    kernel_name: str = "lunarpoles",
) -> None:
    """Execute code cells sequentially in an isolated namespace and write the populated .ipynb file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ns: Dict[str, Any] = {"__name__": "__main__"}
    nb_cells: List[Dict[str, Any]] = []
    exec_count = 1

    print(f"=== Building & Executing {output_path.name} ===")
    for idx, spec in enumerate(cells_spec):
        ctype = spec["type"]
        src = textwrap.dedent(spec["source"]).strip()
        if ctype == "markdown":
            nb_cells.append(make_markdown_cell(src))
            continue

        plt.close("all")
        stdout_buf = io.StringIO()
        old_stdout = sys.stdout
        outputs: List[Dict[str, Any]] = []

        # Strip IPython magic lines like %matplotlib inline before exec()
        clean_lines = [
            line for line in src.splitlines() if not line.strip().startswith("%")
        ]
        exec_src = "\n".join(clean_lines)

        try:
            sys.stdout = stdout_buf
            exec(compile(exec_src, f"<{output_path.name}-cell-{exec_count}>", "exec"), ns)
        except Exception as exc:
            sys.stdout = old_stdout
            tb = traceback.format_exc()
            print(f"ERROR in {output_path.name} cell {exec_count}:\n{tb}")
            raise exc
        finally:
            sys.stdout = old_stdout

        text_out = stdout_buf.getvalue()
        if text_out:
            outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in text_out.splitlines()],
                }
            )

        # Capture any created matplotlib figures
        fig_nums = plt.get_fignums()
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=130)
            img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")
            outputs.append(
                {
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure>"],
                    },
                    "metadata": {},
                    "output_type": "display_data",
                }
            )
        plt.close("all")

        nb_cells.append(make_code_cell(src, exec_count, outputs))
        print(f"  [Cell {exec_count}] executed ({len(outputs)} output blocks)")
        exec_count += 1

    notebook_json = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": f"Python ({kernel_name})",
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.write_text(json.dumps(notebook_json, indent=1) + "\n", encoding="utf-8")
    print(f"=== Saved {output_path} ===\n")
