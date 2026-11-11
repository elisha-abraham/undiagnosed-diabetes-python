"""Build analysis.ipynb from analysis.py and run it.

analysis.py is the source of truth: plain Python with `# %%` cell markers,
so changes show up cleanly in version control. This script turns it into
a notebook, executes every cell, and saves the notebook with its output.

    python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def cells_from(source: str) -> list:
    cells, kind, lines = [], None, []

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code" and text.strip():
            cells.append(nbformat.v4.new_code_cell(text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    return cells


def main() -> None:
    notebook = nbformat.v4.new_notebook(cells=cells_from((ROOT / "analysis.py").read_text(encoding="utf-8")))
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, ROOT / "analysis.ipynb")
    print(f"analysis.ipynb written with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
