"""中文教学 Notebook 的 RLVR、评估、聊天与持久化章节。"""

from textwrap import dedent


CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "source": dedent(source).strip() + "\n"})


def code(source):
    CELLS.append({"cell_type": "code", "source": dedent(source).strip() + "\n"})


md(r'''
## 9. RLVR：从“模仿答案”到“提高高奖励答案的概率”

SFT 的老师给出一条正确回答，交叉熵直接提高它的概率。强化学习先让模型自己抽样，再用奖励评价回答；它并不自动提供正确答案。

| 方法 | 反馈从哪里来 | 本教程是否实现 |
|---|---|---|
| RLHF | 人类偏好通常训练奖励模型，再用强化学习优化策略 | 只解释，不训练奖励模型 |
| RLVR | 程序能验证答案，例如算术答案或代码测试 | 实现严格整数算术奖励 |
| DPO | 对“选中/拒绝”的回答对直接优化偏好目标 | 只解释，不把它当作在线 RL |
| GRPO | 同一问题抽样一组回答，用组内相对奖励替代价值模型 | 实现一次更新的教学简化版 |

“RLVR”描述奖励来源，“GRPO”描述优化方法，二者可以组合。我们先学 REINFORCE 的直觉，再加入概率比、裁剪和参考策略约束。GRPO 的原始方法见 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。下面省略完整生产系统的分布式采样、数据筛选、多次 PPO 更新和复杂奖励组合。

**学习目标：** 能解释“奖励如何改变下一 token 的概率”；能发现没有奖励差异时为何不该声称训练成功。
''')

md(r'''
### 9.1 奖励先写成一个可以单独测试的程序

对题目 `2 + 3 = ?`，回答 `5` 获得 1，其他回答获得 0。本节只接受纯整数，允许首尾空白与正负号；`先猜4再猜5`、`答案是5` 都得 0。这是一个刻意简单的**格式与正确性联合奖励**，不是推理质量评价。

不要用 `eval(answer)` 执行模型文本，也不要从多个数字中挑出碰巧正确的那个。验证器本身就是训练目标的一部分，漏洞会变成可利用的捷径。

预计输出：`[1.0, 1.0, 0.0, 0.0, 0.0]`。自检：空回答必须为 0，含多个候选数字的回答必须为 0。
''')

code(r'''
def arithmetic_reward(answer: str, target: int) -> float:
    # fullmatch 要求整个回答符合一个整数，而非在长文本中找到某个整数。
    match = re.fullmatch(r"\s*([+-]?[0-9]+)\s*", answer)
    if match is None:
        return 0.0
    try:
        value = int(match.group(1))
    except ValueError:
        return 0.0
    return float(value == int(target))


reward_examples = ["5", "  +5\n", "4", "错误0但最后正确5", ""]
print([arithmetic_reward(text, 5) for text in reward_examples])
assert arithmetic_reward("-3", -3) == 1.0
assert arithmetic_reward("5或6", 5) == 0.0
assert arithmetic_reward("答案是5", 5) == 0.0
assert arithmetic_reward("", 0) == 0.0
''')

md(r'''
### 9.2 相对奖励只在同一道题内部比较

同一问题抽样 $G=4$ 个回答，奖励记作 $r_1,\ldots,r_G$。组内优势为

$$A_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)},\qquad
\operatorname{std}(r)=\sqrt{\frac1G\sum_i(r_i-\operatorname{mean}(r))^2}.$$

这里标准差使用总体定义，即 PyTorch 的 `unbiased=False`。`[0, 1, 0, 1]` 的优势是 `[-1, 1, -1, 1]`，不是把不同题目的难度混成一个基线。`[0,0,0,0]` 与 `[1,1,1,1]` 都没有组内相对信号：优势设为 0，并跳过策略更新。

Shape：单组输入 `[G]` → `[G]`；多个问题输入 `[B,G]` → `[B,G]`，每一行独立计算。预计每行优势均值接近 0。练习：预测 `[0,0,0,1]` 的四个优势符号。
''')

code(r'''
def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    # 末维是一道题的回答组，前面的维度不能参与这道题的归一化。
    values = rewards.detach().float()
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("奖励至少需要一个非空组维度")
    centered = values - values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, unbiased=False, keepdim=True)
    safe_std = std.clamp_min(1e-8)
    return torch.where(std > 1e-8, centered / safe_std, torch.zeros_like(centered))


reward_groups = torch.tensor([[0., 1., 0., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]])
advantage_groups = group_advantages(reward_groups)
print("每一行是一道题：\n", advantage_groups)
assert torch.allclose(advantage_groups.mean(dim=-1), torch.zeros(3), atol=1e-6)
assert torch.equal(advantage_groups[1:], torch.zeros(2, 4))
''')

md(r'''
### 9.3 一个一定看得见梯度方向的小实验

语言模型可能暂时一个正确整数都采不到。先用只有两个候选的概率模型观察机制：候选 0 是错误答案，候选 1 是正确答案。两个可训练 logits 经 softmax 得到概率 $p$。

REINFORCE 的样本损失是 $-A\log p(\text{抽到的答案})$。下格人为指定一次“正确答案被抽到”的事件，优势设为 +1，只为了隔离梯度机制。**这不是 TinyLLM 的 rollout，不能把它计入模型正确率或 RL 成果。**

预计：更新前正确答案概率约 0.5，更新后大于 0.5。练习：把优势改为 −1，概率应朝哪个方向移动？为什么奖励/优势必须视作常量？
''')

code(r'''
# 只有两个参数的概率实验，完全不改变已经训练的 model。
toy_logits = nn.Parameter(torch.zeros(2))
toy_optimizer = torch.optim.SGD([toy_logits], lr=0.5)
before = toy_logits.softmax(dim=0).detach().clone()
toy_advantage = torch.tensor(1.0)
toy_loss = -toy_advantage.detach() * toy_logits.log_softmax(dim=0)[1]
toy_optimizer.zero_grad()
toy_loss.backward()
print("logits 的梯度：", toy_logits.grad.tolist())
toy_optimizer.step()
after = toy_logits.softmax(dim=0).detach()
print("更新前概率：", before.tolist())
print("更新后概率：", after.tolist())
assert after[1] > before[1]
del toy_optimizer, toy_logits
''')

md(r'''
### 9.4 哪些 log probability 属于回答？

设 prompt 有 $P$ 个 token，回答有 $T$ 个 token。拼接输入 shape 为 `[1,P+T]`，模型输出 `[1,P+T,V]`。第 `P-1` 个位置的 logits 预测第一个回答 token；选取位置区间 `[P-1, P+T-1)` 后 shape 为 `[T,V]`，再按实际回答 token `gather` 得到 `[T]`。

回答 EOS 也是模型采取的动作，**其 log probability 参与损失**；prompt、padding 不参与。我们直接使用未 padding 的单条序列，因此回答掩码等价于这个切片。EOS 后不再追加任何 token。

下方函数保留梯度；只有旧策略与参考策略的调用放在 `no_grad()` 中。完整词表 `log_softmax` 同时是概率归一化检查：如果 rollout 先做 top-k 截断，却在这里使用完整分布，训练就悄悄换了策略。
''')

code(r'''
def completion_log_probs(policy, prompt_ids, completion_ids) -> torch.Tensor:
    if len(prompt_ids) < 1:
        raise ValueError("prompt 至少含一个 token")
    if len(completion_ids) < 1:
        raise ValueError("回答至少含一个 token；EOS 也算回答 token")
    sequence = list(prompt_ids) + list(completion_ids)
    if len(sequence) > policy.cfg.max_seq_len:
        raise ValueError("prompt 与回答的总长度超过模型上下文")
    policy_device = next(policy.parameters()).device
    ids = torch.tensor([sequence], dtype=torch.long, device=policy_device)
    with amp_context():
        logits, _ = policy(ids, use_cache=False)
    p = len(prompt_ids)
    t = len(completion_ids)
    # logits 的位置落后目标 token 一位；float32 归一化更稳定。
    response_logits = logits[0, p - 1:p + t - 1, :].float()
    targets = torch.tensor(completion_ids, dtype=torch.long, device=policy_device)
    selected = F.log_softmax(response_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    assert selected.shape == (t,)
    return selected


alignment_prompt = encode_prompt("2+3=?")
alignment_completion = tokenizer.encode("5") + [EOS_ID]
alignment_logp = completion_log_probs(model, alignment_prompt, alignment_completion)
print("prompt 长度：", len(alignment_prompt), "回答长度（含EOS）：", len(alignment_completion))
print("回答 log probability shape：", tuple(alignment_logp.shape))
assert alignment_logp.requires_grad
assert torch.isfinite(alignment_logp).all()
del alignment_logp
''')

md(r'''
### 9.5 从 REINFORCE 到带裁剪的策略目标

对于回答中的一个 token，令

$$\rho_t=\exp(\log\pi_\theta(y_t|x,y_{<t})-\operatorname{stopgrad}(\log\pi_{\rm old}(y_t|x,y_{<t}))),$$
$$k_t=\exp(\log\pi_{\rm ref}-\log\pi_\theta)-(\log\pi_{\rm ref}-\log\pi_\theta)-1,$$
$$L=-\operatorname{mean}_t\big[\min(\rho_t A,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A)-\beta k_t\big].$$

旧策略是在抽样时固定下来的概率，参考策略是 RL 开始前的冻结 SFT 模型。这里的 $k_t$ 是一个常见的非负 KL 估计/惩罚项，并非逐个 token 枚举整个词表的精确 KL。若 token 确实从当前策略抽样，它的期望对应 $\mathrm{KL}(\pi_\theta\|\pi_{\rm ref})$；若多次更新后复用旧 rollout，这种解释通常只是近似。不要把一次抽样值直接当成总体 KL。

每条回答先除以自己的 token 数，再平均回答，避免仅因为回答更长而权重更大。教学实现每组只做**一次**优化器更新：第一次前向时 ratio 应约为 1，裁剪机制的作用通过下一格的人工张量自检展示。它是 GRPO 风格的简化策略梯度，不是完整生产版 GRPO。
''')

code(r'''
def grpo_objective(new_logp, old_logp, ref_logp, advantage, clip_eps=0.2, beta=0.02):
    if new_logp.shape != old_logp.shape or new_logp.shape != ref_logp.shape:
        raise ValueError("新、旧、参考 log probability 的 shape 必须相同")
    if new_logp.ndim not in (1, 2) or new_logp.shape[-1] == 0:
        raise ValueError("输入应为 [T] 或未 padding 的 [B,T]")
    # 旧策略和参考策略不参与反向传播；奖励优势也不参与求导。
    current = new_logp.float()
    old = old_logp.detach().float()
    reference = ref_logp.detach().float()
    adv = torch.as_tensor(advantage, dtype=current.dtype, device=current.device).detach()
    while adv.ndim < current.ndim:
        adv = adv.unsqueeze(-1)
    ratio = torch.exp(current - old)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    ref_minus_new = reference - current
    sampled_kl = torch.exp(ref_minus_new) - ref_minus_new - 1.0
    # 末维先按回答长度归一化，再对组内回答平均。
    return (-surrogate + beta * sampled_kl).mean(dim=-1).mean()


# 故意制造 ratio=2：正优势的裁剪目标最多到 1.2。
test_new = torch.tensor([math.log(0.4)], requires_grad=True)
test_old = torch.tensor([math.log(0.2)], requires_grad=True)
test_ref = torch.tensor([math.log(0.4)], requires_grad=True)
test_loss = grpo_objective(test_new, test_old, test_ref, 1.0, beta=0.0)
test_loss.backward()
print("裁剪后的损失（预计 -1.2）：", float(test_loss.detach()))
assert torch.allclose(test_loss.detach(), torch.tensor(-1.2), atol=1e-6)
assert test_old.grad is None and test_ref.grad is None
assert torch.allclose(test_new.grad, torch.zeros_like(test_new))
''')

md(r'''
### 9.6 真实 rollout：抽四次，按回答逐条反向传播

下方 rollout 固定温度为 1，不做 top-k、top-p 或特殊 token 屏蔽。它与 `completion_log_probs` 的完整 softmax 是同一个策略。出现 PAD/BOS/角色 token 的回答奖励为 0，避免把解码器隐藏的特殊 token 当成合法整数。结束 EOS 保留在概率序列中，解码用于奖励时再移除。

抽样和评分都在 `eval()` 模式下完成，更新前向也保持同模式；`eval()` 不会关闭梯度，`no_grad()` 才会。这让含 dropout 的扩展模型也不会在采样/评分间换分布。每条回答立即 `backward()`，不堆四条计算图；旧 log probability 只保存张量，不额外复制一份旧模型。复用前章的 GradScaler 接口：fp16 缩放梯度，bf16/CPU 禁用缩放；先 unscale，再裁剪，最后 step/update，溢出时明确记录跳过。

一组奖励零方差时返回 `skipped=True`，不编造正确回答来强行更新。参考 SFT 模型需要一份冻结副本；8 GB 显存的 full 模式仍应从短上下文、单条微批开始。代码只保留这一份参考模型，RL 结束即释放。
''')

code(r'''
@torch.no_grad()
def _rl_rollout(policy, prompt_ids, max_new_tokens=8):
    # 使用完整词表概率抽样，故意不应用聊天时常见的截断策略。
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须至少为1")
    limit = min(int(max_new_tokens), policy.cfg.max_seq_len - len(prompt_ids))
    if limit < 1:
        raise ValueError("prompt 没有为回答留下上下文空间")
    policy_device = next(policy.parameters()).device
    sequence = list(prompt_ids)
    completion = []
    for _ in range(limit):
        ids = torch.tensor([sequence], dtype=torch.long, device=policy_device)
        with amp_context():
            logits, _ = policy(ids, use_cache=False)
        probabilities = F.softmax(logits[0, -1].float(), dim=-1)
        token = int(torch.multinomial(probabilities, num_samples=1).item())
        completion.append(token)
        sequence.append(token)
        if token == EOS_ID:
            break
    return completion


def _arithmetic_completion(completion_ids, target):
    # 角色或padding不能被decode隐藏后伪装成有效答案。
    forbidden = {PAD_ID, BOS_ID, USER_ID, ASSISTANT_ID, SYSTEM_ID}
    invalid_special = any(token in forbidden for token in completion_ids)
    answer = tokenizer.decode([token for token in completion_ids if token != EOS_ID]).strip()
    reward = 0.0 if invalid_special else arithmetic_reward(answer, target)
    return answer, reward, invalid_special


def rlvr_step(policy, reference, question, target, optimizer, group_size=4, max_new_tokens=8, scaler=None):
    if group_size < 2:
        raise ValueError("相对奖励至少需要两个回答")
    prompt_ids = encode_prompt(question)
    was_training = policy.training
    policy.eval()
    reference.eval()
    optimizer.zero_grad(set_to_none=True)
    try:
        completions = [_rl_rollout(policy, prompt_ids, max_new_tokens) for _ in range(group_size)]
        scored = [_arithmetic_completion(ids, target) for ids in completions]
        policy_device = next(policy.parameters()).device
        rewards = torch.tensor([entry[1] for entry in scored], device=policy_device)
        advantages = group_advantages(rewards)
        statistics = {
            "question": question, "target": int(target),
            "answers": [entry[0] for entry in scored], "rewards": rewards.cpu().tolist(),
            "advantages": advantages.cpu().tolist(),
            "mean_reward": float(rewards.mean().item()),
            "response_tokens": [len(ids) for ids in completions],
            "empty_answers": sum(entry[0] == "" for entry in scored),
            "invalid_special_answers": sum(entry[2] for entry in scored),
            "skipped": False, "skip_reason": None, "loss": None,
        }
        if rewards.std(unbiased=False).item() <= 1e-8:
            statistics.update(skipped=True, skip_reason="组内奖励零方差，没有相对学习信号")
            return statistics
        scaler = make_scaler() if scaler is None else scaler
        # 在任何参数更新前缓存抽样策略概率，detach 保证 old 固定。
        with torch.no_grad():
            old_logps = [completion_log_probs(policy, prompt_ids, ids).detach() for ids in completions]
            ref_logps = [completion_log_probs(reference, prompt_ids, ids).detach() for ids in completions]
        loss_total = 0.0
        for ids, old_logp, ref_logp, advantage in zip(completions, old_logps, ref_logps, advantages):
            new_logp = completion_log_probs(policy, prompt_ids, ids)
            # 每条微批的权重是1/G；本条的计算图在backward后释放。
            loss = grpo_objective(new_logp, old_logp, ref_logp, advantage) / group_size
            if not torch.isfinite(loss).item():
                raise FloatingPointError("RL损失出现NaN/Inf；本次更新停止，请检查精度和概率数值")
            scaler.scale(loss).backward()
            loss_total += float(loss.detach().item())
            del new_logp, loss
        # 与前章保持同样顺序：unscale → clip → step → update。
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        if not scaler.is_enabled() and not torch.isfinite(grad_norm).item():
            raise FloatingPointError("RL梯度出现NaN/Inf；未启用缩放器，本次更新停止")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        successful = scaler.get_scale() >= previous_scale and torch.isfinite(grad_norm).item()
        statistics.update(loss=loss_total, grad_norm=float(grad_norm.item()),
                          skipped=not successful,
                          skip_reason=None if successful else "AMP检测到非有限梯度，跳过参数更新")
        return statistics
    finally:
        optimizer.zero_grad(set_to_none=True)
        policy.train(was_training)
''')

md(r'''
### 9.7 两次尝试只验证训练链路

本格默认 smoke 与 full **都只尝试 2 次 RL 更新**。从随机初始化、很少的预训练/SFT 出发，四次采样全得 0 很正常。日志若显示 `跳过`，含义就是“这次没有更新”，不是暗中注入标准答案后的成功。若你要研究能力提升，需自行增加高质量训练数据、SFT 收敛程度、抽样数和训练步数，并另建固定验证集。

保存两个明确标记的检查点：`sft_before_rl` 与 `rlvr_after_attempts`。第二个记录实际更新数，可能为 0。不能仅凭文件名、一次随机正确回答或训练 loss 下降断言 RL 有效。

显存提示：0.1B 的 float32 权重约 0.4 GB；policy + reference 约 0.8 GB，Adam 两个状态约 0.8 GB，加上梯度约 0.4 GB，再加激活与临时张量。实际峰值取决于上下文与实现；此处释放前面阶段的优化器状态后再开始 RL。
''')

code(r'''
import copy


def save_stage_checkpoint(stage, policy, metadata=None):
    checkpoint_dir = WORK_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{PROFILE}_{stage}.pt"
    torch.save({
        "stage": stage, "profile": PROFILE, "config": asdict(policy.cfg),
        "state_dict": {name: value.detach().cpu() for name, value in policy.state_dict().items()},
        "metadata": metadata or {},
    }, checkpoint_path)
    return checkpoint_path


SFT_BEFORE_RL_PATH = save_stage_checkpoint("sft_before_rl", model, {"rl_updates": 0})
# 前面的训练优化器是函数内局部变量；重复执行本格时只释放本节自身对象。
for _name in ("rl_optimizer", "rl_scaler", "reference_model"):
    if _name in globals():
        del globals()[_name]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
reference_model = copy.deepcopy(model).eval().requires_grad_(False)
rl_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
rl_scaler = make_scaler()
RL_STEPS = 2 if PROFILE == "smoke" else 2
RL_TRAIN_CASES = [
    (f"请只给出整数结果：{a} + {b} 等于多少？", a + b)
    for a, b in [(2, 3), (5, 4), (7, 2), (1, 8)]
]
print("SFT 检查点：", SFT_BEFORE_RL_PATH)
print("计划 RL 尝试：", RL_STEPS, "，每题采样4条；并不保证产生更新")
''')

code(r'''
rl_history = []
for rl_index in range(RL_STEPS):
    question, target = RL_TRAIN_CASES[rl_index % len(RL_TRAIN_CASES)]
    stats = rlvr_step(model, reference_model, question, target, rl_optimizer,
                      group_size=4, max_new_tokens=8, scaler=rl_scaler)
    rl_history.append(stats)
    print(f"尝试 {rl_index + 1}/{RL_STEPS}：reward={stats['rewards']}，"
          f"{'跳过' if stats['skipped'] else '已更新'}，loss={stats['loss']}")
    if stats["skipped"]:
        print("跳过原因：", stats["skip_reason"])
    print("真实模型回答：", [repr(answer) for answer in stats["answers"]])
rl_updates = sum(not row["skipped"] for row in rl_history)
RL_AFTER_PATH = save_stage_checkpoint("rlvr_after_attempts", model, {
    "attempts": RL_STEPS, "rl_updates": rl_updates, "group_size": 4,
    "history": rl_history, "note": "教学短训练；不代表能力提升",
})
print(f"实际更新数：{rl_updates}/{RL_STEPS}；检查点：{RL_AFTER_PATH}")
# 后面的评估与聊天只需一个模型，不长期占用参考模型与优化器状态。
del reference_model, rl_optimizer, rl_scaler
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
''')

md(r'''
## 10. 评估：把一次输出与稳定能力分开

这里验证**本教程合成训练集未使用的算术问题**。默认教学数据的训练操作数是 `0..9`；下面只使用 `31..79` 的操作数，RL 训练题也未包含它们。full 的外部语料未必满足这个约定，因此下格扫描实际预训练/SFT train 文本中的算式，排除粗检发现的重叠并显示排除数。真实语料需要独立、完整去污染：这个表达式粗检不保证模型未见过题目的改写、答案或等价形式。验证答案不能追加回训练集后继续称为 heldout。

指标至少同时报告：`correct / total`、空回答数量、含非法特殊 token 的数量。两组随机种子各抽样一次，用来观察采样变化；这是每个种子的单样本正确率，不是 best-of-N、多数投票或 pass@k。smoke 的零分是正常且有意义的结果，说明跑通链路与训练出能力是不同目标。

自检：精确训练问题交集应为空；粗检排除后必须按实际剩余题数统计，全部排除时不报告正确率。
''')

code(r'''
HELDOUT_PAIRS = [(31, 47), (52, 36), (64, 35), (79, 32), (43, 58), (68, 37),
                 (35, 62), (74, 39), (56, 38), (71, 46), (49, 63), (67, 54)]
heldout_count = 4 if PROFILE == "smoke" else len(HELDOUT_PAIRS)
HELDOUT_CASES = [(f"请只给出整数结果：{a} + {b} 等于多少？", a + b)
                 for a, b in HELDOUT_PAIRS[:heldout_count]]
synthetic_training_questions = set(question for question, _ in RL_TRAIN_CASES)
synthetic_training_questions.update(message["content"] for record in demo_sft_records
                                    for message in record["messages"] if message["role"] == "user")
assert not synthetic_training_questions.intersection(question for question, _ in HELDOUT_CASES)
# 检查前文实际读入的SFT训练记录，而非只检查本节的RL题目。
training_questions = {message["content"] for record in sft_train_records
                      for message in record["messages"] if message["role"] == "user"}
exact_question_overlap = training_questions.intersection(question for question, _ in HELDOUT_CASES)
# 同时检查预训练原文中a+b和b+a两种写法；逐篇流式扫描，不堆全部文本。
heldout_patterns = [(a, b, re.compile(rf"(?<![0-9])(?:{a}\s*\+\s*{b}|{b}\s*\+\s*{a})\s*(?:=|等于)"))
                    for a, b in HELDOUT_PAIRS[:heldout_count]]
heldout_contamination = set()
for a, b in HELDOUT_PAIRS[:heldout_count]:
    if f"请只给出整数结果：{a} + {b} 等于多少？" in exact_question_overlap:
        heldout_contamination.add((a, b))
for training_text in iter_split_text(split_paths["train"]):
    for a, b, pattern in heldout_patterns:
        if pattern.search(training_text):
            heldout_contamination.add((a, b))
for record in sft_train_records:
    for message in record["messages"]:
        for a, b, pattern in heldout_patterns:
            if pattern.search(message["content"]):
                heldout_contamination.add((a, b))
HELDOUT_CASES = [(f"请只给出整数结果：{a} + {b} 等于多少？", a + b)
                 for a, b in HELDOUT_PAIRS[:heldout_count] if (a, b) not in heldout_contamination]
print("表达式粗检排除题数：", len(heldout_contamination), "；剩余验证题数：", len(HELDOUT_CASES))


@torch.no_grad()
def evaluate_arithmetic(policy, cases, seeds=(2027, 2028), max_new_tokens=8):
    result_rows = []
    policy_device = next(policy.parameters()).device
    cuda_devices = ([policy_device.index if policy_device.index is not None else torch.cuda.current_device()]
                    if policy_device.type == "cuda" else [])
    was_training = policy.training
    policy.eval()
    try:
        for seed in seeds:
            records = []
            # 不让评估的种子改变Notebook后续的随机数状态。
            with torch.random.fork_rng(devices=cuda_devices):
                # CPU参考评估只设置CPU种子，避免顺便改变其他GPU的随机数状态。
                torch.random.default_generator.manual_seed(seed)
                if policy_device.type == "cuda":
                    with torch.cuda.device(policy_device):
                        torch.cuda.manual_seed(seed)
                for question, target in cases:
                    completion = _rl_rollout(policy, encode_prompt(question), max_new_tokens)
                    answer, reward, invalid = _arithmetic_completion(completion, target)
                    records.append({"question": question, "target": target, "answer": answer,
                                    "correct": int(reward), "empty": int(answer == ""),
                                    "invalid_special": int(invalid)})
            total = len(records)
            correct = sum(row["correct"] for row in records)
            result_rows.append({"seed": seed, "correct": correct, "total": total,
                                "accuracy": correct / total if total else None,
                                "empty": sum(row["empty"] for row in records),
                                "invalid_special": sum(row["invalid_special"] for row in records),
                                "records": records})
    finally:
        policy.train(was_training)
    return result_rows


heldout_results = evaluate_arithmetic(model, HELDOUT_CASES)
for row in heldout_results:
    if row["total"] == 0:
        print(f"seed={row['seed']}：没有剩余验证题，不报告正确率。请另建独立验证集。")
        continue
    print(f"seed={row['seed']}：正确 {row['correct']}/{row['total']}，"
          f"accuracy={row['accuracy']:.1%}，空回答={row['empty']}，"
          f"非法特殊token={row['invalid_special']}")
    print("第一题抽样回答：", repr(row["records"][0]["answer"]))
''')

md(r'''
### 10.1 多轮聊天也是 token 拼接与上下文预算

消息结构为 `[{"role":"user","content":"你好"}, ...]`，最后一条必须是 user。先完整编码历史消息与当前问题，再追加 ASSISTANT 标记，模型随后续写回答。

不能无限追加历史：先为 `max_new_tokens` 预留空间，超过预算时删除最早的**完整 user/assistant 轮次**，尽量保留最近完整轮次。若 system + 当前 user 本身已过长，直接提示缩短；不把一个汉字的部分字节或半个角色标记当作完整问题。代码返回被保留的消息供自检，`chat_once` 本身只返回回答，不偷偷修改调用者的历史。

本教程只调用一次示例，不默认运行无限 `input()` 循环。smoke 输出乱码或空字符串很正常；一个从头开始、只训练几步的模型还没有学会中文对话。
''')

code(r'''
def build_chat_prompt(messages, max_new_tokens=32):
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("聊天消息最后一条必须是user")
    if not 1 <= int(max_new_tokens) < model.cfg.max_seq_len:
        raise ValueError("回答长度必须大于0并小于上下文长度")
    copied = [{"role": message["role"], "content": str(message["content"])} for message in messages]
    system_messages = copied[:1] if copied[0]["role"] == "system" else []
    middle = copied[len(system_messages):-1]
    if len(middle) % 2 or any(middle[i]["role"] != ("user" if i % 2 == 0 else "assistant")
                              for i in range(len(middle))):
        raise ValueError("历史需要由完整user/assistant轮次组成，system只允许位于开头")
    rounds = [middle[i:i + 2] for i in range(0, len(middle), 2)]
    current_user = [copied[-1]]

    def encode_selected(selected):
        # SFT函数要求以assistant结束；空占位消息仅用于构造相同格式的前缀。
        ids, _ = encode_sft(selected + [{"role": "assistant", "content": ""}])
        # 删除空回答的EOS，使prompt恰好停在ASSISTANT标记后。
        return list(ids[:-1])

    budget = model.cfg.max_seq_len - int(max_new_tokens)
    minimal = system_messages + current_user
    if len(encode_selected(minimal)) > budget:
        raise ValueError("system与当前问题过长；请缩短文本或减少max_new_tokens")
    selected = system_messages + [message for pair in rounds for message in pair] + current_user
    while len(encode_selected(selected)) > budget and rounds:
        rounds.pop(0)
        selected = system_messages + [message for pair in rounds for message in pair] + current_user
    prompt_ids = encode_selected(selected)
    assert len(prompt_ids) + int(max_new_tokens) <= model.cfg.max_seq_len
    return prompt_ids, selected


def chat_once(messages, max_new_tokens=32):
    prompt_ids, _ = build_chat_prompt(messages, max_new_tokens)
    generated = generate(model, prompt_ids, max_new_tokens=max_new_tokens,
                         temperature=0.8, top_k=None, top_p=0.95, use_cache=True)
    completion = generated[len(prompt_ids):]
    if EOS_ID in completion:
        completion = completion[:completion.index(EOS_ID)]
    special_ids = {PAD_ID, BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID, SYSTEM_ID}
    return tokenizer.decode([token for token in completion if token not in special_ids]).strip()


chat_messages = [{"role": "user", "content": "你好"},
                 {"role": "assistant", "content": "你好！"},
                 {"role": "user", "content": "什么是注意力？"}]
chat_prompt_ids, retained_messages = build_chat_prompt(chat_messages, max_new_tokens=16)
chat_answer = chat_once(chat_messages, max_new_tokens=16)
print("保留消息数：", len(retained_messages), "prompt长度：", len(chat_prompt_ids))
print("模型聊天答复：", repr(chat_answer))
print("提示：smoke文字混乱或空回答是短训练的预期现象。")
assert len(chat_prompt_ids) + 16 <= model.cfg.max_seq_len
assert len(chat_messages) == 3
''')

md(r'''
### 10.2 KV cache 应该改变计算量，不能改变模型语义

不使用 cache 时，每生成一个 token 都重新计算整个前缀的 K/V。使用 cache 时，保留过去 K/V，只算新 token，并正确使用它的绝对位置。先比较“同一前缀、同一新 token”的最后 logits，再比较 top-k=1 的输出序列；这比仅观察一次计时更能发现缓存位置与掩码错误。

预计：最大 logits 差异很小，两种生成输出一致。GPU 混合精度允许小数值差异；短序列的 cache 未必更快，Python 调度开销可能占主要部分。计时只是当前环境的一次观察，不是性能结论。练习：画出无 cache 与 cache 的 K/V shape 在每一步如何变化。
''')

code(r'''
cache_prompt = encode_prompt("你好")
cache_token = tokenizer.encode("好")[0]
cache_device = next(model.parameters()).device
model.eval()
with torch.no_grad(), amp_context():
    prefix_tensor = torch.tensor([cache_prompt], dtype=torch.long, device=cache_device)
    _, prefix_cache = model(prefix_tensor, use_cache=True)
    step_logits, _ = model(torch.tensor([[cache_token]], dtype=torch.long, device=cache_device),
                           use_cache=True, kv_cache=prefix_cache)
    full_logits, _ = model(torch.tensor([cache_prompt + [cache_token]], dtype=torch.long,
                                       device=cache_device), use_cache=False)
logit_error = (step_logits[0, -1].float() - full_logits[0, -1].float()).abs().max().item()
torch.testing.assert_close(step_logits[0, -1].float(), full_logits[0, -1].float(),
                           atol=5e-2, rtol=1e-2)
print("缓存与完整前向的最大logits误差：", logit_error)


def timed_greedy(use_cache):
    if cache_device.type == "cuda":
        torch.cuda.synchronize(cache_device)
    started = time.perf_counter()
    # top_k=1保留唯一最高概率token，因此两次运行没有采样差异。
    generated = generate(model, cache_prompt, max_new_tokens=4 if PROFILE == "smoke" else 16,
                         temperature=1.0, top_k=1, top_p=1.0, use_cache=use_cache)
    if cache_device.type == "cuda":
        torch.cuda.synchronize(cache_device)
    return generated, time.perf_counter() - started


without_cache, seconds_without = timed_greedy(False)
with_cache, seconds_with = timed_greedy(True)
assert without_cache == with_cache
print(f"无缓存 {seconds_without:.4f}s；有缓存 {seconds_with:.4f}s；输出token一致")
del prefix_cache, step_logits, full_logits
''')

md(r'''
### 10.3 保存、重载与继续练习

一个可重载的推理包至少有：权重 `state_dict`、模型配置、tokenizer 的 JSON，以及训练阶段说明。仅保存权重而丢失特殊 token ID/词表，会让同一整数序列被解码成不同文本。

下格保存当前模型，元数据写明 `PROFILE`、RL 尝试与实际更新数。重载函数创建一个**新变量**并返回，默认不调用，不覆盖当前训练好的 `model`。推理包不含优化器状态；精确续训还需保存优化器、调度器、数据进度与随机数状态。

自检：检查配置中的 `vocab_size` 等于重载 tokenizer 的词表大小；特殊 token ID 必须一致。权威材料可继续阅读 [DeepSeekMath](https://arxiv.org/abs/2402.03300)，并把本文的单次更新与论文完整实验区分开。
''')

code(r'''
BUNDLE_DIR = WORK_DIR / f"{PROFILE}_inference_bundle"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()},
           BUNDLE_DIR / "model_state.pt")
(BUNDLE_DIR / "config.json").write_text(json.dumps(asdict(model.cfg), ensure_ascii=False, indent=2),
                                        encoding="utf-8")
# 底层Tokenizers直接保存JSON，保留byte-level规则、词表与特殊token约定。
tokenizer.backend.save(str(BUNDLE_DIR / "tokenizer.json"))
(BUNDLE_DIR / "training_stage.json").write_text(json.dumps({
    "profile": PROFILE, "stage": "rlvr_after_attempts", "rl_attempts": RL_STEPS,
    "rl_updates": rl_updates,
    "special_token_ids": {name: tokenizer.backend.token_to_id(name) for name in SPECIAL_TOKENS},
    "description": "从头训练的教学模型；短训练不代表对话或算术能力",
}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_inference_bundle(bundle_dir, device="cpu"):
    bundle_dir = Path(bundle_dir)
    saved_config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    restored_tokenizer = TextTokenizer(Tokenizer.from_file(str(bundle_dir / "tokenizer.json")))
    saved_stage = json.loads((bundle_dir / "training_stage.json").read_text(encoding="utf-8"))
    for name, expected_id in saved_stage["special_token_ids"].items():
        if restored_tokenizer.backend.token_to_id(name) != expected_id:
            raise ValueError(f"特殊token {name} 的ID与保存时不同")
    restored_cfg = ModelConfig(**saved_config)
    if restored_cfg.vocab_size != restored_tokenizer.vocab_size:
        raise ValueError("配置与tokenizer的词表大小不一致")
    restored_model = TinyLLM(restored_cfg)
    state = torch.load(bundle_dir / "model_state.pt", map_location="cpu", weights_only=True)
    restored_model.load_state_dict(state, strict=True)
    restored_model.to(device).eval()
    return restored_model, restored_tokenizer, restored_cfg


print("已保存推理包：", BUNDLE_DIR.resolve())
print("文件：", sorted(path.name for path in BUNDLE_DIR.iterdir()))
# 需要重载时手动运行下一行，使用新变量且优先CPU，避免覆盖或显存双份占用。
# reloaded_model, reloaded_tokenizer, reloaded_cfg = load_inference_bundle(BUNDLE_DIR, device="cpu")
''')

md(r'''
### 10.4 可选：用同一验证集比较 SFT 与 RL 后状态

比较必须用相同题目、相同采样种子、相同解码策略，以及同一设备与精度。下面开关默认为 False；打开后，把 SFT 检查点加载到临时 CPU 模型，同时将当前模型暂移 CPU，对两个状态都重新评估。完成后释放临时模型，并把当前模型移回原设备，不替换它的权重。full 的 CPU 评估会慢。

可能看到两阶段都是 0 分，也可能看到少量随机差别。这些小样本不能证明改进；尤其若 `rl_updates=0`，两阶段权重完全相同，同种子结果也应相同。

**下一步练习（按顺序）：**

1. 用自己的文字推导 `P-1` 为什么预测第一个回答 token，并画出 EOS 的位置。
2. 把正确答案概率实验改成负优势，验证概率下降；再把优势设成 0，验证不更新。
3. 在另一份数据副本中增加 SFT，直到模型确实能采到部分正确整数，再观察组内奖励差异。
4. 固定训练/验证/测试三份数据，增加种子与样本量，比较 SFT 和 RL；记录算力、更新数、奖励分布。
5. 若加入特殊 token 屏蔽或温度，先让 rollout 与 log probability 使用相同分布，再谈性能。

完整学习路线是：小张量弄清索引 → 跑通 smoke → 独立验证模型结构 → 增加数据与训练 → 用 heldout 判断能力。选择 full 只把模型规模提升到约 0.1B，不会自动增加数据质量或训练时长。
''')

code(r'''
RUN_STAGE_COMPARISON = False
if RUN_STAGE_COMPARISON:
    sft_checkpoint = torch.load(SFT_BEFORE_RL_PATH, map_location="cpu", weights_only=True)
    comparison_model = TinyLLM(ModelConfig(**sft_checkpoint["config"]))
    comparison_model.load_state_dict(sft_checkpoint["state_dict"], strict=True)
    comparison_model.eval()
    del sft_checkpoint
    original_device = next(model.parameters()).device
    original_training = model.training
    model.to("cpu").eval()
    try:
        sft_results = evaluate_arithmetic(comparison_model, HELDOUT_CASES)
        comparison_after = evaluate_arithmetic(model, HELDOUT_CASES)
        for before, after in zip(sft_results, comparison_after):
            print(f"seed={before['seed']}：SFT={before['correct']}/{before['total']}，"
                  f"RL后={after['correct']}/{after['total']}，实际RL更新={rl_updates}")
    finally:
        model.to(original_device).train(original_training)
    del comparison_model
    gc.collect()
else:
    print("阶段比较未运行；将RUN_STAGE_COMPARISON改为True后可执行同种子对照。")
''')
