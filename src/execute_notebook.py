from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_PATH = ROOT / "notebooks" / "01_data_quality_and_harmonization.ipynb"


def main() -> None:
    notebook_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else DEFAULT_NOTEBOOK_PATH
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace = {"__name__": "__main__"}
    execution_count = 0
    original_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                    exec(
                        compile(cell.source, f"{notebook_path.name}:cell_{cell_index + 1}", "exec"),
                        namespace,
                    )
                text = stdout.getvalue()
                cell.outputs = (
                    [nbformat.v4.new_output("stream", name="stdout", text=text)] if text else []
                )
            except Exception as error:
                text = stdout.getvalue()
                traceback_lines = traceback.format_exc().splitlines()
                outputs = []
                if text:
                    outputs.append(nbformat.v4.new_output("stream", name="stdout", text=text))
                outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(error).__name__,
                        evalue=str(error),
                        traceback=traceback_lines,
                    )
                )
                cell.outputs = outputs
                nbformat.write(notebook, notebook_path)
                raise
    finally:
        os.chdir(original_cwd)
    nbformat.write(notebook, notebook_path)
    print(f"Executed {execution_count} code cells: {notebook_path}")


if __name__ == "__main__":
    main()
