"""执行教学短实验、运行关键行为验证，可另测完整模型的一步 CUDA 训练。"""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "0.1B_LLM.ipynb"


def load_notebook_definitions(path):
    """载入离线基础小实验与所有定义；跳过预训练/SFT/RL 的训练调用。"""
    notebook = nbformat.read(path, as_version=4)
    ns = {"__name__": "notebook_validation"}
    old_profile = os.environ.get("LLM_PROFILE")
    old_artifact_dir = os.environ.get("LLM_ARTIFACT_DIR")
    temporary = tempfile.TemporaryDirectory(prefix="llm_notebook_checks_")
    # 保持临时目录的生存期，测试 checkpoint 时 tokenizer.json 仍存在。
    ns["_validation_temporary_directory"] = temporary
    os.environ["LLM_PROFILE"] = "smoke"
    os.environ["LLM_ARTIFACT_DIR"] = temporary.name
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            section = cell.metadata.get("section", "")
            if section in {"foundations", "model"}:
                selected_source = cell.source
            else:
                tree = ast.parse(cell.source)
                selected = []
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        selected.append(node)
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                        try:
                            ast.literal_eval(node.value)
                        except (ValueError, TypeError, SyntaxError):
                            continue
                        selected.append(node)
                if not selected:
                    continue
                selected_source = ast.unparse(ast.Module(body=selected, type_ignores=[]))
            # 基础实验输出由报告器收集，不淹没行为测试结果。
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(selected_source, f"notebook-cell-{index+1}", "exec"), ns)
        return ns
    finally:
        if old_profile is None:
            os.environ.pop("LLM_PROFILE", None)
        else:
            os.environ["LLM_PROFILE"] = old_profile
        if old_artifact_dir is None:
            os.environ.pop("LLM_ARTIFACT_DIR", None)
        else:
            os.environ["LLM_ARTIFACT_DIR"] = old_artifact_dir


def full_cuda_probe(ns):
    """验证完整98.3M模型能完成真实前向、反向和AdamW更新，不做长期训练。"""
    torch = ns["torch"]
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA 不可用"}
    # 清除基础 smoke 模型，仅为测试腾出空间。
    ns["model"].cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = ns["TinyLLM"](ns["FULL_CFG"]).cuda().train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, foreach=False)
    ids = torch.randint(0, policy.cfg.vocab_size, (1, 257), device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with ns["amp_context"]():
        logits, _ = policy(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, policy.cfg.vocab_size), ids[:, 1:].reshape(-1))
    scaler = torch.amp.GradScaler("cuda", enabled=ns["AMP_DTYPE"] == torch.float16)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    assert torch.isfinite(loss) and torch.isfinite(grad_norm)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    result = {
        "status": "passed", "parameters": sum(p.numel() for p in policy.parameters()),
        "gpu": torch.cuda.get_device_name(0), "context": 256, "micro_batch": 1,
        "loss": loss.item(), "grad_norm": grad_norm.item(),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "seconds": round(time.perf_counter() - started, 3),
        "scope": "随机token上的一次更新，仅验证实现和资源，不是语言能力评测",
    }
    del optimizer, policy, loss, logits, ids
    torch.cuda.empty_cache()
    return result


def cleanup_notebook_namespace(ns):
    """Windows 下先关闭 memmap 文件句柄，再清理验证临时目录。"""
    for name in ("train_tokens", "val_tokens", "test_tokens"):
        array = ns.get(name)
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    temporary = ns.get("_validation_temporary_directory")
    if temporary is not None:
        temporary.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="真实Jupyter内核顺序执行默认smoke并保存输出")
    parser.add_argument("--full-probe", action="store_true", help="另测98.3M模型的一步CUDA更新")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    report = {"notebook": NOTEBOOK.name, "cells": len(notebook.cells), "syntax": "passed"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell-{index+1}")
    if args.execute:
        os.environ["LLM_PROFILE"] = "smoke"
        started = time.perf_counter()
        kernel_name = os.environ.get("LLM_NOTEBOOK_KERNEL", notebook.metadata.kernelspec.name)
        client = NotebookClient(notebook, timeout=600, kernel_name=kernel_name, resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.write(notebook, NOTEBOOK)
        report["smoke_execution"] = {"status": "passed", "seconds": round(time.perf_counter()-started,3),
                                     "code_cells": sum(c.cell_type == "code" for c in notebook.cells)}
        print("完整 smoke Notebook 已逐格执行。", flush=True)
    ns = load_notebook_definitions(NOTEBOOK)
    spec = importlib.util.spec_from_file_location("notebook_tests", ROOT / "tests/test_notebook.py")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    try:
        report["behavior_checks"] = test_module.run_behavior_checks(ns)
        for check in report["behavior_checks"]:
            print(check["status"], check["name"], check.get("detail", ""), flush=True)
        if args.full_probe:
            report["full_cuda_probe"] = full_cuda_probe(ns)
            print("完整模型一步验证:", json.dumps(report["full_cuda_probe"], ensure_ascii=False), flush=True)
    finally:
        cleanup_notebook_namespace(ns)
    output = ROOT / "validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告:", output)
    return 1 if any(c["status"] == "failed" for c in report["behavior_checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
