"""Notebook 的独立行为验收；仅用小模型，避免测试触发下载或长训练。

验证器把顺序执行后的命名空间传给 ``run_behavior_checks(namespace)``。
独立运行：``python tests/test_notebook.py``；验证器须公开
``load_notebook_definitions(path) -> dict``，仅加载可安全执行的定义。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "0.1B_LLM.ipynb"


def _need(ns, name):
    assert name in ns, f"Notebook 未公开必需定义：{name}"
    return ns[name]


def _model(ns, *, use_sdpa=True):
    torch = _need(ns, "torch")
    torch.manual_seed(731)
    cfg = _need(ns, "ModelConfig")(
        vocab_size=64, dim=32, n_layers=2, n_heads=4,
        hidden_dim=64, max_seq_len=64,
        use_sdpa=use_sdpa, gradient_checkpointing=False,
    )
    return _need(ns, "TinyLLM")(cfg).cpu()


def _forward(model, ids, **kwargs):
    result = model(ids, **kwargs)
    assert isinstance(result, tuple) and len(result) == 2, (
        "TinyLLM.forward 应始终返回 (logits, kv_cache)"
    )
    return result


def _ids(value):
    """把 tokenizer / dataset 的序列统一为 Python list，保留实际 token。"""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if hasattr(value, "ids"):
        value = value.ids
    return list(value)


def _special_id(ns, name):
    if name.upper() in ns:
        return int(ns[name.upper()])
    tokenizer = _need(ns, "tokenizer")
    assert hasattr(tokenizer, name.lower()), f"须公开 {name.upper()} 或 tokenizer.{name.lower()}"
    return int(getattr(tokenizer, name.lower()))


def _sft_example(ns, question="2 + 3 = ?", answer="5"):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    encoded = _need(ns, "encode_sft")(messages)
    assert isinstance(encoded, tuple) and len(encoded) == 2
    return messages, _ids(encoded[0]), _ids(encoded[1])


def _assert_nested_equal(torch, left, right, path="optimizer"):
    """检查真实 Adam 状态，避免仅验证恢复后模型权重而遗漏优化器。"""
    if torch.is_tensor(left):
        assert torch.is_tensor(right), f"{path}: 类型不一致"
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), f"{path}: keys 不一致"
        for key in left:
            _assert_nested_equal(torch, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), f"{path}: 长度不一致"
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_nested_equal(torch, a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def check_full_model_parameter_count(ns):
    torch = _need(ns, "torch")
    cfg = _need(ns, "FULL_CFG")
    # meta 设备只建立形状；验收参数预算不必分配约 400 MB 的 float32 权重。
    with torch.device("meta"):
        model = _need(ns, "TinyLLM")(cfg)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 98_324_736, f"完整配置参数数为 {count:,}，预期 98,324,736"
    assert cfg.dim == 768 and cfg.n_layers == 13 and cfg.vocab_size == 8192


def check_token_windows_predict_next_token(ns):
    torch = _need(ns, "torch")
    tokens = torch.arange(19, dtype=torch.long)
    dataset = _need(ns, "TokenWindowDataset")(tokens, seq_len=5)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert _ids(x) == [0, 1, 2, 3, 4]
    assert _ids(y) == [1, 2, 3, 4, 5], "预训练 target 必须恰好向前移一位"


def check_causal_prefix_is_unchanged(ns):
    torch = _need(ns, "torch")
    model = _model(ns).eval()
    first = torch.tensor([[1, 4, 7, 9, 11, 12]])
    second = torch.tensor([[1, 4, 7, 31, 32, 33]])
    with torch.no_grad():
        logits_a, _ = _forward(model, first)
        logits_b, _ = _forward(model, second)
    torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(logits_a[:, 3:], logits_b[:, 3:]), (
        "测试输入应使后缀改变，避免因输出恒定而误通过"
    )


def check_manual_attention_matches_sdpa(ns):
    torch = _need(ns, "torch")
    manual = _model(ns, use_sdpa=False).eval()
    sdpa = _model(ns, use_sdpa=True).eval()
    sdpa.load_state_dict(manual.state_dict())
    tokens = torch.tensor([[1, 2, 3, 7, 11, 5], [4, 8, 2, 6, 9, 3]])
    with torch.no_grad():
        expected, _ = _forward(manual, tokens)
        actual, _ = _forward(sdpa, tokens)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def check_cache_matches_single_token_and_chunk_forward(ns):
    torch = _need(ns, "torch")
    tokens = torch.tensor([[1, 2, 4, 8, 16, 31, 9], [3, 6, 12, 24, 7, 14, 28]])
    # 两条 attention 实现都必须正确处理缓存的绝对位置和非方形因果 mask。
    for use_sdpa in (False, True):
        model = _model(ns, use_sdpa=use_sdpa).eval()
        with torch.no_grad():
            expected, _ = _forward(model, tokens)
            for widths in ([1] * 7, [3, 2, 2]):
                cache, outputs, start = None, [], 0
                for width in widths:
                    chunk = tokens[:, start:start + width]
                    logits, cache = _forward(model, chunk, use_cache=True, kv_cache=cache)
                    assert cache is not None and len(cache) == 2
                    start += width
                    for key, value in cache:
                        assert key.shape == value.shape
                        assert key.shape == (2, 4, start, 8), "KV cache 应是 [B,H,T,Dh]"
                    outputs.append(logits)
                actual = torch.cat(outputs, dim=1)
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def check_generate_cache_preserves_tokens_and_training_mode(ns):
    model = _model(ns).train()
    generate = _need(ns, "generate")
    prompt = [1, 4, 7]
    cached = generate(model, prompt, max_new_tokens=5, temperature=0, use_cache=True)
    assert model.training, "generate应恢复调用前的train/eval状态"
    uncached = generate(model, prompt, max_new_tokens=5, temperature=0, use_cache=False)
    assert cached == uncached and cached[:len(prompt)] == prompt
    assert len(prompt) < len(cached) <= len(prompt) + 5
    unchanged = generate(model, prompt, max_new_tokens=0, temperature=0)
    assert unchanged == prompt and model.training


def check_chat_prompt_matches_sft_and_trims_whole_rounds(ns):
    build_prompt = _need(ns, "build_chat_prompt")
    single = [{"role": "user", "content": "你好"}]
    prompt, retained = build_prompt(single, max_new_tokens=8)
    assert _ids(prompt) == _ids(_need(ns, "encode_prompt")("你好"))
    assert retained == single and single == [{"role": "user", "content": "你好"}]
    # 旧轮次明显超出上下文：只能一起移除user/assistant，保留system和当前问题。
    system = {"role": "system", "content": "简短回答。"}
    current = {"role": "user", "content": "你好"}
    messages = [system, {"role": "user", "content": "旧问题。" * 400},
                {"role": "assistant", "content": "旧回答。" * 400}, current]
    prompt, retained = build_prompt(messages, max_new_tokens=8)
    assert retained == [system, current], "超长历史应按完整轮次裁剪"
    assert len(messages) == 4, "聊天构造不能偷偷修改调用者消息列表"
    assert len(prompt) + 8 <= _need(ns, "model").cfg.max_seq_len


def check_sft_prompt_mask_and_eos_supervision(ns):
    _, ids, labels = _sft_example(ns)
    prompt = _ids(_need(ns, "encode_prompt")("2 + 3 = ?"))
    eos_id = _special_id(ns, "eos_id")
    assert len(ids) == len(labels) and len(ids) > len(prompt)
    assert ids[:len(prompt)] == prompt, "训练 prompt 必须与推理 prompt 完全相同"
    assert labels[:len(prompt)] == [-100] * len(prompt), "用户文字与角色前缀不应参与 SFT loss"
    assert labels[len(prompt):] == ids[len(prompt):], "回答 token 应受到监督"
    assert ids[-1] == labels[-1] == eos_id, "EOS 也要监督，模型才有机会学会结束回答"
    assert labels[len(prompt)] != -100, "第一个回答 token 不应被误遮罩"


def check_sft_shift_and_right_padding(ns):
    torch = _need(ns, "torch")
    messages_a, ids_a, labels_a = _sft_example(ns, "2 + 3 = ?", "5")
    messages_b, ids_b, labels_b = _sft_example(ns, "Please calculate 13 + 24 = ?", "37")
    dataset = _need(ns, "SFTDataset")([
        {"messages": messages_a}, {"messages": messages_b},
    ])
    item_a, item_b = dataset[0], dataset[1]
    assert _ids(item_a[0]) == ids_a[:-1]
    assert _ids(item_a[1]) == labels_a[1:], "SFTDataset 应移位一次，训练器不再移位"
    assert _ids(item_b[0]) == ids_b[:-1] and _ids(item_b[1]) == labels_b[1:]
    x, y = _need(ns, "sft_collate")([item_a, item_b])
    assert torch.is_tensor(x) and torch.is_tensor(y)
    assert x.shape == y.shape == (2, max(len(ids_a), len(ids_b)) - 1)
    pad_id = _special_id(ns, "pad_id")
    for row, (ids, labels) in enumerate(((ids_a, labels_a), (ids_b, labels_b))):
        width = len(ids) - 1
        assert x[row, :width].tolist() == ids[:-1]
        assert y[row, :width].tolist() == labels[1:]
        assert bool((x[row, width:] == pad_id).all())
        assert bool((y[row, width:] == -100).all()), "padding 不能变成监督目标"
    prompt_len = len(_ids(_need(ns, "encode_prompt")("2 + 3 = ?")))
    assert int(y[0, prompt_len - 1]) == ids_a[prompt_len], (
        "最后一个 prompt 位置的 logits 应预测第一个 answer token"
    )


def check_token_loss_uses_already_shifted_targets(ns):
    torch = _need(ns, "torch")
    loss_sum = _need(ns, "token_loss_sum")
    targets = torch.tensor([[-100, 3, 5, -100]])
    logits = torch.zeros((1, 4, 8))
    logits[0, 1, 3] = 20
    logits[0, 2, 5] = 20
    summed, count = loss_sum(logits, targets)
    assert int(count) == 2 and float(summed) < 1e-5, (
        "token_loss_sum 接收已shift targets，不能再次将标签错开"
    )
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 8), targets.reshape(-1), ignore_index=-100, reduction="sum",
    )
    torch.testing.assert_close(summed, expected)


def check_all_ignored_targets_have_finite_zero_loss(ns):
    torch = _need(ns, "torch")
    logits = torch.randn((2, 4, 8), requires_grad=True)
    targets = torch.full((2, 4), -100, dtype=torch.long)
    summed, count = _need(ns, "token_loss_sum")(logits, targets)
    assert int(count) == 0 and torch.isfinite(summed) and float(summed.detach()) == 0
    summed.backward()
    assert logits.grad is not None and bool((logits.grad == 0).all()), (
        "全ignore batch需可微的0，并由训练器按有效token数守卫，不能除以0"
    )


def check_reward_rejects_ambiguous_or_partial_answers(ns):
    reward = _need(ns, "arithmetic_reward")
    assert reward("1", 1) == 1 and reward("  -12\n", -12) == 1
    for answer in ("0", "错误0但最后正确1", "0 or 1", "1 and 2", "11", "1.0", "1e0", "", "= 1"):
        assert reward(answer, 1) == 0, f"奖励必须严格匹配唯一整数答案，错误接受了：{answer!r}"


def check_group_advantages_are_per_prompt(ns):
    torch = _need(ns, "torch")
    rewards = torch.tensor([[0., 1., 1., 0.], [8., 9., 11., 12.], [1., 1., 1., 1.]])
    advantages = _need(ns, "group_advantages")(rewards)
    assert advantages.shape == rewards.shape and bool(torch.isfinite(advantages).all())
    torch.testing.assert_close(advantages.mean(dim=-1), torch.zeros(3), atol=1e-6, rtol=0)
    assert bool((advantages[-1] == 0).all()), "同组奖励无差异时 advantage 应全为0"
    expected = (rewards[:2] - rewards[:2].mean(dim=-1, keepdim=True)) / rewards[:2].std(
        dim=-1, unbiased=False, keepdim=True,
    )
    torch.testing.assert_close(advantages[:2], expected, rtol=1e-4, atol=1e-5)


def check_completion_log_probs_match_causal_prediction(ns):
    torch = _need(ns, "torch")
    model = _model(ns).eval()
    prompt, completion = [1, 2, 3], [4, 5, 6, 7]
    actual = _need(ns, "completion_log_probs")(model, prompt, completion)
    assert actual.shape == (len(completion),), "completion_log_probs应仅返回回答token概率"
    assert actual.requires_grad, "新策略的 log-prob 必须保留梯度"
    logits, _ = _forward(model, torch.tensor([prompt + completion]))
    predicted = logits[0, len(prompt) - 1:-1].log_softmax(dim=-1)
    expected = predicted.gather(-1, torch.tensor(completion).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    (-actual.mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def check_grpo_old_and_reference_are_detached(ns):
    torch = _need(ns, "torch")
    new = torch.tensor([[-1.1, -1.3, -1.0], [-0.8, -1.2, -1.4]], requires_grad=True)
    old = torch.tensor([[-1.2, -1.0, -1.1], [-0.9, -1.0, -1.3]], requires_grad=True)
    reference = torch.tensor([[-1.0, -1.4, -1.2], [-1.1, -1.3, -1.1]], requires_grad=True)
    advantages = torch.tensor([1., -1.])
    loss = _need(ns, "grpo_objective")(
        new, old, reference, advantages, clip_eps=0.2, beta=0.02,
    )
    assert torch.is_tensor(loss) and loss.ndim == 0 and bool(torch.isfinite(loss))
    ratio = (new - old.detach()).exp()
    advantage = advantages.unsqueeze(-1)
    expected_policy = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    delta = reference.detach() - new
    expected_kl = delta.exp() - delta - 1
    expected = (expected_policy + 0.02 * expected_kl).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    loss.backward()
    assert new.grad is not None and bool(torch.isfinite(new.grad).all())
    assert float(new.grad.abs().sum()) > 0
    assert old.grad is None and reference.grad is None, "old/ref必须在目标函数内部detach"


def check_checkpoint_restores_logits_optimizer_step_and_sampler(ns):
    torch = _need(ns, "torch")
    model = _model(ns)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    # 用CPU上的真实缩放器检验非空scale/growth状态，不额外占用CUDA显存。
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.0, growth_interval=2)
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    summed, count = _need(ns, "token_loss_sum")(_forward(model, inputs)[0], targets)
    scaler.scale(summed / count).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        expected, _ = _forward(model, inputs)
    sampler = torch.Generator().manual_seed(987)
    history = [{"step": 7, "loss": 1.25}]
    with tempfile.TemporaryDirectory(prefix="tinyllm_checkpoint_") as temp_dir:
        path = Path(temp_dir) / "state.pt"
        _need(ns, "save_checkpoint")(
            path, model, optimizer, scaler, step=7, history=history, sampler_generator=sampler,
        )
        expected_random = torch.rand(5, generator=sampler)
        restored_model = _model(ns).eval()
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.1)
        restored_sampler = torch.Generator().manual_seed(123)
        restored_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=16.0, growth_interval=7)
        payload = _need(ns, "load_checkpoint")(
            path, restored_model, restored_optimizer, restored_scaler, sampler_generator=restored_sampler,
        )
        assert payload["step"] == 7 and payload["history"] == history
        with torch.no_grad():
            actual, _ = _forward(restored_model, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_nested_equal(torch, optimizer.state_dict(), restored_optimizer.state_dict())
        _assert_nested_equal(torch, scaler.state_dict(), restored_scaler.state_dict(), "scaler")
        torch.testing.assert_close(torch.rand(5, generator=restored_sampler), expected_random, rtol=0, atol=0)
        # 再做一步相同更新：恢复的 Adam 动量应导致完全相同的结果。
        for current_model, current_optimizer, current_scaler in (
            (model, optimizer, scaler), (restored_model, restored_optimizer, restored_scaler),
        ):
            current_model.train()
            summed, count = _need(ns, "token_loss_sum")(_forward(current_model, inputs)[0], targets)
            current_scaler.scale(summed / count).backward()
            current_scaler.step(current_optimizer)
            current_scaler.update()
            current_optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(_forward(model, inputs)[0], _forward(restored_model, inputs)[0], rtol=0, atol=0)
        _assert_nested_equal(torch, scaler.state_dict(), restored_scaler.state_dict(), "updated_scaler")


def check_short_overfit_reduces_loss(ns):
    torch = _need(ns, "torch")
    model = _model(ns).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.0)
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]]).repeat(2, 1)
    x, y = ids[:, :-1], ids[:, 1:]
    losses = []
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = _forward(model, x)
        summed, count = _need(ns, "token_loss_sum")(logits, y)
        loss = summed / count
        assert bool(torch.isfinite(loss)), "短程训练出现非有限loss"
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.5, (
        f"确定性小批量应能过拟合：初始loss={losses[0]:.4f}，最终loss={losses[-1]:.4f}"
    )


CHECKS: tuple[Callable, ...] = (
    check_full_model_parameter_count,
    check_token_windows_predict_next_token,
    check_causal_prefix_is_unchanged,
    check_manual_attention_matches_sdpa,
    check_cache_matches_single_token_and_chunk_forward,
    check_generate_cache_preserves_tokens_and_training_mode,
    check_chat_prompt_matches_sft_and_trims_whole_rounds,
    check_sft_prompt_mask_and_eos_supervision,
    check_sft_shift_and_right_padding,
    check_token_loss_uses_already_shifted_targets,
    check_all_ignored_targets_have_finite_zero_loss,
    check_reward_rejects_ambiguous_or_partial_answers,
    check_group_advantages_are_per_prompt,
    check_completion_log_probs_match_causal_prediction,
    check_grpo_old_and_reference_are_detached,
    check_checkpoint_restores_logits_optimizer_step_and_sampler,
    check_short_overfit_reduces_loss,
)


def run_behavior_checks(ns) -> list[dict]:
    """全部检查都运行；单项失败会保留栈信息，不阻断其余检查。"""
    results = []
    torch = ns.get("torch")
    previous_threads = torch.get_num_threads() if torch is not None else None
    if torch is not None:
        torch.set_num_threads(1)
    try:
        for check in CHECKS:
            start = time.perf_counter()
            try:
                check(ns)
                result = {"name": check.__name__, "status": "passed", "detail": ""}
            except Exception as error:
                result = {
                    "name": check.__name__, "status": "failed",
                    "detail": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            result["seconds"] = round(time.perf_counter() - start, 4)
            results.append(result)
    finally:
        if torch is not None:
            torch.set_num_threads(previous_threads)
    return results


def _load_namespace(notebook_path):
    assert notebook_path.is_file(), (
        f"Notebook 尚未生成：{notebook_path.name}；此失败是交付前的预期 RED 验证"
    )
    validator_path = ROOT / "tools" / "validate_notebook.py"
    assert validator_path.is_file(), "尚未实现 tools/validate_notebook.py"
    spec = importlib.util.spec_from_file_location("tinyllm_notebook_validator", validator_path)
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    assert hasattr(validator, "load_notebook_definitions"), (
        "validator 须提供 load_notebook_definitions(path) -> namespace"
    )
    return validator.load_notebook_definitions(notebook_path)


def main():
    # Windows 控制台也输出 UTF-8，保留中文断言和报告的可读性。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        ns = _load_namespace(NOTEBOOK_PATH)
    except AssertionError as error:
        print(f"FAIL: {error}")
        return 1
    try:
        results = run_behavior_checks(ns)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return int(any(result["status"] != "passed" for result in results))
    finally:
        # 关闭Windows上的memmap句柄；run_behavior_checks本身不清理共享ns。
        validator = sys.modules["tinyllm_notebook_validator"]
        validator.cleanup_notebook_namespace(ns)


if __name__ == "__main__":
    raise SystemExit(main())
