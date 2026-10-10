"""从分章 cell 源码构建独立可运行的教学 Notebook。"""
import importlib
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python (pytorch)", "language": "python", "name": "pytorch"},
        "language_info": {"name": "python", "version": "3.11", "file_extension": ".py"},
        "tutorial": {"language": "zh-CN", "target_parameters": 98324736, "default_profile": "smoke"},
    }
    for section in ("foundations", "model", "training", "rl_chat", "next_steps"):
        module = importlib.import_module(f"tools.notebook_parts.{section}")
        for index, spec in enumerate(module.CELLS):
            factory = nbformat.v4.new_markdown_cell if spec["cell_type"] == "markdown" else nbformat.v4.new_code_cell
            cell = factory(spec["source"])
            cell.id = f"{section}-{index:03d}"
            cell.metadata["section"] = section
            notebook.cells.append(cell)
    nbformat.validate(notebook)
    path = ROOT / "0.1B_LLM.ipynb"
    nbformat.write(notebook, path)
    return path, notebook


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path, notebook = build()
    print(f"已生成 {path}；{len(notebook.cells)} cells，{sum(c.cell_type == 'code' for c in notebook.cells)} code cells。")
