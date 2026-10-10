"""教学 Notebook 的第 7、8 章；仅导出单元格，不执行训练。"""

from textwrap import dedent

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "source": dedent(source).strip()})


def code(source):
    CELLS.append({"cell_type": "code", "source": dedent(source).strip()})


md(r"""
## 7. 预训练：把“预测下一个 token”变成一次参数更新

前面我们已经得到分词器、三份 token 数据和 `TinyLLM`。这一章只解决一个问题：**怎样用文本教模型预测下一个 token？**

学习顺序是：移位样本 → 有效 token 损失 → 学习率 → 验证 → 保存与恢复 → 短训练。运行默认的 20 步 smoke 只验证流程；它不代表模型已经学会语言。约 0.1B 参数的正式配置也需要足够的高质量数据和训练 token。

本章约定：`steps` 始终表示成功执行的 **optimizer updates**；`grad_accum` 个小批次才组成一次更新。输入和目标的形状均为 `[B,T]`，模型输出为 `[B,T,V]`。
""")

md(r"""
### 7.1 为什么目标要右移一位？

给定窗口 `[我, 喜欢, 学习, EOS]`，输入是 `[我, 喜欢, 学习]`，目标是 `[喜欢, 学习, EOS]`。因果掩码使位置 `t` 只能看见 `0..t`，目标是 `t+1`，因此不会偷看答案。

\[
\mathcal L=-\frac{1}{N}\sum_{b,t\in\text{有效位置}}\log p_\theta(y_{b,t}\mid x_{b,0:t}).
\]

`TokenWindowDataset` 把索引解释为窗口起点；随机性由 sampler 提供。它只复制当前长度 `T+1` 的窗口，适合读取磁盘上的 `np.memmap`，不用把整份语料搬进内存。
""")

code(r"""
class TokenWindowDataset(Dataset):
    def __init__(self, tokens, seq_len):
        # 至少需要 T+1 个 token，最后一个 token 只出现在目标中。
        self.tokens = tokens
        self.seq_len = int(seq_len)
        if self.seq_len < 1 or len(tokens) <= self.seq_len:
            raise ValueError("token 数据必须至少包含 seq_len+1 个元素")

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, start):
        start = int(start)
        if not 0 <= start < len(self):
            raise IndexError(start)
        # 显式复制只读 memmap 切片，转换成 Embedding 需要的 int64。
        window = torch.from_numpy(
            np.array(self.tokens[start:start + self.seq_len + 1], dtype=np.int64)
        )
        return window[:-1], window[1:]

# 可观察自检：同一份窗口在输入、目标中只错开一个位置。
toy_window = TokenWindowDataset(np.arange(9), seq_len=4)
toy_x, toy_y = toy_window[2]
assert toy_x.tolist() == [2, 3, 4, 5]
assert toy_y.tolist() == [3, 4, 5, 6]
print("输入：", toy_x.tolist(), "目标：", toy_y.tolist())
""")

md(r"""
### 7.2 随机训练窗口与固定验证窗口

训练窗口可以重复采样，这是从长 token 流随机截取片段的简单办法。验证采用固定、均匀分布的起点，便于重复比较；验证文本必须来自先前隔离的 `val_tokens`，不能从训练 token 流中截取。

显存由**小批大小**和 `T` 影响，而有效批量还取决于梯度累计。若 `B=1,T=256,K=16`，一次更新处理约 `4096` 个目标 token。Windows 的 Notebook 使用 `num_workers=0`，避免工作进程导入 Notebook 状态的问题。
""")

code(r"""
from torch.utils.data import RandomSampler, Subset

PT_BATCH_SIZE = int(os.environ.get("LLM_PT_BATCH_SIZE", "4" if PROFILE == "smoke" else "1"))
PT_GRAD_ACCUM = int(os.environ.get("LLM_PT_GRAD_ACCUM", "2" if PROFILE == "smoke" else "16"))
pretrain_dataset = TokenWindowDataset(train_tokens, cfg.max_seq_len)
pretrain_sampler_generator = torch.Generator().manual_seed(2026)
pretrain_sampler = RandomSampler(
    pretrain_dataset, replacement=True,
    num_samples=min(len(pretrain_dataset), 8192),
    generator=pretrain_sampler_generator,
)
pretrain_loader = DataLoader(
    pretrain_dataset, batch_size=PT_BATCH_SIZE, sampler=pretrain_sampler,
    num_workers=0, pin_memory=DEVICE.type == "cuda",
)

# 验证起点固定；按 token 数加权的损失函数会在下一节实现。
val_dataset = TokenWindowDataset(val_tokens, cfg.max_seq_len)
val_count = min(len(val_dataset), 128 if PROFILE == "smoke" else 1024)
val_starts = np.linspace(0, len(val_dataset) - 1, val_count, dtype=np.int64).tolist()
pretrain_val_loader = DataLoader(
    Subset(val_dataset, val_starts), batch_size=PT_BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=DEVICE.type == "cuda",
)
batch_x, batch_y = next(iter(pretrain_loader))
assert batch_x.shape == batch_y.shape
assert torch.equal(batch_x[:, 1:], batch_y[:, :-1])
print("小批形状：", tuple(batch_x.shape), "每次更新约含 token：",
      PT_BATCH_SIZE * cfg.max_seq_len * PT_GRAD_ACCUM)
""")

md(r"""
### 7.3 损失和 AdamW 各自负责什么？

交叉熵把 logits `[B,T,V]` 展开成 `[B*T,V]`，目标展开成 `[B*T]`。`-100` 表示不参与损失的目标；预训练窗口没有 padding，但这个约定会贯穿 SFT。

我们先求损失之和与有效 token 数，再由调用者做除法。这样长短不等的小批不会获得相同权重。AdamW 更新参数，并对矩阵权重使用 weight decay；偏置、归一化向量和其他一维参数不衰减。
""")

code(r"""
def token_loss_sum(logits, targets):
    # targets 已由 Dataset 移位；这里绝对不能再移位一次。
    total = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    count = int((targets != -100).sum().item())
    return total, count


def make_optimizer(model, lr):
    matrices, vectors = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (matrices if parameter.ndim >= 2 else vectors).append(parameter)
    return torch.optim.AdamW(
        [{"params": matrices, "weight_decay": 0.1},
         {"params": vectors, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )

# 可观察自检：均匀分布预测的交叉熵应当是 log(V)。
uniform_logits = torch.zeros(2, 3, 7, requires_grad=True)
masked_targets = torch.tensor([[1, 2, -100], [0, -100, 4]])
uniform_sum, uniform_count = token_loss_sum(uniform_logits, masked_targets)
assert uniform_count == 4
assert abs(uniform_sum.item() / uniform_count - math.log(7)) < 1e-6
print("有效目标：", uniform_count, "平均损失：", uniform_sum.item() / uniform_count)
""")

md(r"""
### 7.4 为什么先 warmup，再慢慢降低学习率？

刚开始时参数和 Adam 的统计量尚不稳定，先从小学习率增加到峰值；随后用余弦曲线降低步长。以下 `update_index` 从 `0` 开始计数，代表下一次 optimizer update。

\[
\eta(s)=\eta_{\max}\frac{s+1}{W}\quad(s<W),\qquad
\eta(s)=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}(1+\cos(\pi q)).
\]

其中 `q` 是 warmup 之后的归一化进度。短演示的默认 warmup 占 10%；正式训练时应根据总更新数设置。它是教学默认值，不是对所有语料都最优的超参数。
""")

code(r"""
def scheduled_lr(update_index, total_steps, peak_lr, warmup_steps, min_ratio=0.1):
    if update_index < warmup_steps:
        return peak_lr * (update_index + 1) / max(1, warmup_steps)
    progress = (update_index - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return peak_lr * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2)

lr_preview = [scheduled_lr(s, 20, 3e-4, 2) for s in range(20)]
assert abs(lr_preview[1] - 3e-4) < 1e-12
assert abs(lr_preview[-1] - 3e-5) < 1e-12
plt.figure(figsize=(6, 2.5))
plt.plot(range(1, 21), lr_preview, marker=".")
plt.xlabel("optimizer update")
plt.ylabel("learning rate")
plt.grid(alpha=0.3)
plt.show()
""")

md(r"""
### 7.5 验证损失怎样公平地平均？

假设两个验证批分别包含 `4` 和 `1` 个有效 token，平均损失分别为 `2` 和 `8`。正确总体平均是 `(4*2+1*8)/5=3.2`，不是 `(2+8)/2=5`。

验证不更新参数；暂时切换为 `eval`，结束后恢复原模式。困惑度为 `exp(loss)`，仅在同一分词器、同一验证集和相同掩码约定下可比较。`max_batches` 可让教学试跑更快，但限制后的指标只代表所评估的子集。
""")

code(r"""
@torch.no_grad()
def evaluate_lm(model, loader, max_batches=None):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for batch_index, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with amp_context():
                logits, _ = model(inputs, use_cache=False)
                loss_sum, token_count = token_loss_sum(logits, targets)
            total_loss += float(loss_sum.item())
            total_tokens += token_count
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("验证集没有任何有效目标 token")
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss,
            "perplexity": math.exp(mean_loss) if mean_loss < 700 else float("inf"),
            "tokens": total_tokens}

# 一次短验证建立训练前基线；结果通常接近 log(vocab_size)，无需强求精确相等。
pretrain_baseline = evaluate_lm(model, pretrain_val_loader, max_batches=4)
print("训练前验证：", pretrain_baseline)
""")

md(r"""
### 7.6 梯度累计与 AMP：一次更新有四个步骤

1. 先取得 `K` 个微批，并数出它们一共含 `N` 个有效目标。
2. 每个微批反向传播 `loss_sum/N`，得到所有有效 token 的平均梯度。
3. 若使用 CUDA fp16，先 `unscale_`，再裁剪梯度，最后尝试更新参数。
4. 清空梯度，为下一次更新开始累计。

AMP 在前向中使用合适的精度；参数和 AdamW 状态仍保持通常的 float32。CUDA bf16 一般不需要缩放器，CPU 路径关闭缩放。**不能先裁剪缩放后的梯度，也不能把不等长微批的均值直接除以 K。**

\[
\nabla\mathcal L=\sum_{k=1}^{K}\frac{N_k}{\sum_j N_j}\nabla\overline{\mathcal L}_k.
\]
""")

code(r"""
def make_scaler():
    # 只有 fp16 CUDA 路径开启缩放；bf16 和 CPU 仍使用同一调用接口。
    return torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)


def train_one_update(model, micro_batches, optimizer, scaler):
    total_tokens = sum(int((targets != -100).sum().item()) for _, targets in micro_batches)
    if total_tokens == 0:
        raise ValueError("一次更新没有任何有效目标 token")
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for inputs, targets in micro_batches:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        with amp_context():
            logits, _ = model(inputs, use_cache=False)
            loss_sum, _ = token_loss_sum(logits, targets)
            loss = loss_sum / total_tokens
        if not torch.isfinite(loss).item():
            raise FloatingPointError("损失出现 NaN/Inf；请检查数据、学习率和精度")
        scaler.scale(loss).backward()
        total_loss += float(loss_sum.detach().item())
    # 顺序必须是：还原梯度 → 裁剪 → optimizer step → 更新 scale。
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not scaler.is_enabled() and not torch.isfinite(grad_norm).item():
        raise FloatingPointError("未启用 scaler 时梯度出现 NaN/Inf，停止本次更新")
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    successful = scaler.get_scale() >= old_scale
    optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / total_tokens, "tokens": total_tokens,
            "grad_norm": float(grad_norm.item()), "successful": successful}
""")

md(r"""
### 7.7 为什么 checkpoint 不只是一个权重文件？

继续训练还需要 AdamW 的一阶、二阶统计量、AMP scale、成功更新数、历史记录与随机状态。还要确认当前模型配置、分词器 SHA256 和数据指纹与保存时一致，避免把同一个 token ID 解释成不同文本。

下面先写临时文件，`flush + fsync` 后用 `os.replace` 替换目标文件。检查点仅包含 Tensor、字符串、数字、列表和字典等安全类型；NumPy 随机状态转为普通数值与 Tensor，因此可以用 `torch.load(..., weights_only=True)` 读取。

**恢复的边界：**保存 RNG 和采样 generator 便于继续训练，但没有保存当前 DataLoader 迭代器中尚未消费的索引。重新创建迭代器后，样本次序可能不同；本教程不承诺逐批精确重放。CUDA 内核也不一定逐位确定。
""")

code(r"""
import hashlib


def checkpoint_metadata(model):
    model_config = getattr(model, "cfg", getattr(model, "config", cfg))
    # 数据指纹在数据准备章计算，包含三份 bin 与分词器的内容哈希。
    tokenizer_sha = hashlib.sha256((WORK_DIR / "tokenizer.json").read_bytes()).hexdigest()
    summary = {name: {"tokens": int(len(values)), "dtype": str(values.dtype)}
               for name, values in [("train", train_tokens), ("val", val_tokens), ("test", test_tokens)]}
    return {"config": asdict(model_config), "tokenizer_sha256": tokenizer_sha,
            "data_fingerprint": str(DATA_FINGERPRINT), "data_summary": summary}


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": {"algorithm": numpy_state[0],
                      "keys": torch.tensor(numpy_state[1].astype(np.int64)),
                      "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])}}


def restore_rng_state(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"],
                         numpy_state["keys"].cpu().numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"],
                         numpy_state["cached_gaussian"]))


def save_checkpoint(path, model, optimizer, scaler, step, history, sampler_generator=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, **checkpoint_metadata(model),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scaler": scaler.state_dict(), "step": int(step), "history": history,
               "rng": capture_rng_state(),
               "sampler_generator": None if sampler_generator is None else sampler_generator.get_state(),
               "stage_data_summary": getattr(model, "_training_data_summary", None),
               "training_config": getattr(model, "_training_run_config", {})}
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
""")

code(r"""
def load_checkpoint(path, model, optimizer=None, scaler=None, sampler_generator=None):
    # 保持 weights_only=True，不为不兼容的文件自动退回任意 pickle 执行。
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"format_version", "config", "tokenizer_sha256", "data_fingerprint", "data_summary",
                "model", "optimizer", "scaler", "step", "history", "rng", "sampler_generator"}
    if not required.issubset(checkpoint) or checkpoint["format_version"] != 1:
        raise ValueError("不支持的 checkpoint 格式")
    expected = checkpoint_metadata(model)
    # 先检查兼容性，再修改模型与 optimizer，防止部分加载。
    for key in ["config", "tokenizer_sha256", "data_fingerprint", "data_summary"]:
        if checkpoint[key] != expected[key]:
            raise ValueError(f"checkpoint 与当前 {key} 不一致")
    if hasattr(model, "_training_data_summary"):
        if checkpoint.get("stage_data_summary") != model._training_data_summary:
            raise ValueError("checkpoint 与本阶段训练/验证数据摘要不一致")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    if sampler_generator is not None and checkpoint["sampler_generator"] is not None:
        sampler_generator.set_state(checkpoint["sampler_generator"].cpu())
    return checkpoint


def loader_generator(loader):
    # RandomSampler 的 generator 控制随机窗口；SFT shuffle 也使用该接口。
    return getattr(loader.sampler, "generator", None) or loader.generator


def describe_loader_data(loader):
    dataset = loader.dataset
    description = {"examples": int(len(dataset))}
    if isinstance(dataset, Subset):
        description["subset_indices"] = [int(index) for index in dataset.indices]
        dataset = dataset.dataset
    if hasattr(dataset, "records_sha256"):
        description.update({"records_sha256": dataset.records_sha256,
                            "dropped": int(dataset.dropped)})
    if hasattr(dataset, "seq_len"):
        description["seq_len"] = int(dataset.seq_len)
    return description
""")

md(r"""
### 7.8 把小函数串成训练循环

`train_lm(..., steps=S)` 的含义是训练到总计 `S` 次成功更新。恢复时从 checkpoint 的 `step` 接着计数；例如文件保存了 `12` 次更新，`steps=20` 还会运行 `8` 次。

循环只做调度：获取微批 → 设置本次学习率 → 调用一次更新 → 定期验证、保存。fp16 梯度溢出时 scaler 会跳过参数更新；这次不增加 `step`，并限制连续跳过次数以免无限循环。历史中的 `tokens` 是该次成功更新处理的有效目标数。

恢复必须保持阶段、峰值学习率和累计设置一致；如果增大 `steps`，未来的 warmup/余弦计划会随总目标重新计算，这不是原计划的逐位延续。
""")

code(r"""
def train_lm(model, loader, val_loader, steps, lr=3e-4, grad_accum=1,
             stage="pretrain", resume_from=None):
    steps, grad_accum = int(steps), int(grad_accum)
    if steps < 1 or grad_accum < 1 or lr <= 0 or len(loader) == 0:
        raise ValueError("steps、grad_accum、lr 和训练 loader 大小必须为正")
    optimizer, scaler = make_optimizer(model, lr), make_scaler()
    sampler_generator = loader_generator(loader)
    model._training_data_summary = {"train": describe_loader_data(loader),
                                    "validation": describe_loader_data(val_loader)}
    history, step = [], 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scaler, sampler_generator)
        saved_run = checkpoint.get("training_config", {})
        for key, value in [("stage", stage), ("lr", float(lr)), ("grad_accum", grad_accum)]:
            if key in saved_run and saved_run[key] != value:
                raise ValueError(f"恢复训练时 {key} 必须与 checkpoint 一致")
        history, step = list(checkpoint["history"]), int(checkpoint["step"])
        print(f"恢复到 update={step}；新目标为 {steps}，采样顺序不保证精确重放")
    warmup_steps = max(1, int(steps * 0.1))
    model._training_run_config = {"stage": stage, "lr": float(lr), "grad_accum": grad_accum,
                                  "steps": steps, "warmup_steps": warmup_steps}
    eval_every = max(1, int(os.environ.get("LLM_EVAL_EVERY", max(1, steps // 4))))
    eval_max_batches = 8 if PROFILE == "smoke" else 20
    iterator = iter(loader)
    skipped = 0
    total_tokens = sum(int(item["tokens"]) for item in history)
    model.train()
    while step < steps:
        micro_batches = []
        for _ in range(grad_accum):
            try:
                micro_batches.append(next(iterator))
            except StopIteration:
                iterator = iter(loader)
                micro_batches.append(next(iterator))
        current_lr = scheduled_lr(step, steps, lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        started = time.perf_counter()
        update = train_one_update(model, micro_batches, optimizer, scaler)
        if not update.pop("successful"):
            skipped += 1
            print("AMP 跳过溢出的更新；当前 scale：", scaler.get_scale())
            if skipped >= 20:
                raise FloatingPointError("连续跳过 20 次更新，请降低学习率或改用 bf16")
            continue
        skipped = 0
        step += 1
        total_tokens += update["tokens"]
        record = {"step": step, **update, "lr": current_lr, "total_tokens": total_tokens,
                  "seconds": time.perf_counter() - started}
        if step % eval_every == 0 or step == steps:
            validation = evaluate_lm(model, val_loader, max_batches=eval_max_batches)
            record["val_loss"] = validation["loss"]
            record["val_tokens"] = validation["tokens"]
        history.append(record)
        if "val_loss" in record:
            save_checkpoint(WORK_DIR / f"{stage}_last.pt", model, optimizer, scaler,
                            step, history, sampler_generator)
            print(f"[{stage}] update {step}/{steps}: train={record['loss']:.4f}, "
                  f"val={record['val_loss']:.4f}, lr={current_lr:.2e}")
    return history
""")

md(r"""
### 7.9 现在运行一次短预训练

默认 smoke 和 full 都为 `20` 次短更新，不会自动开始长训练。可以在运行本章前设置 `LLM_PT_STEPS`、`LLM_PT_BATCH_SIZE` 和 `LLM_PT_GRAD_ACCUM`。full 运行需先准备前章要求的本地正式语料。

RTX 4060 Laptop 的 8 GiB 显存建议 full 从 `batch_size=1` 开始，借助前章的梯度检查点和本章的累计控制显存。出现 OOM 时先减小序列长度或微批，并重新建立兼容的配置与数据 loader。**短试训的成功标准是流程完整、损失有限、文件可恢复；不强求每一批损失下降。**
""")

code(r"""
PT_STEPS = int(os.environ.get("LLM_PT_STEPS", "20"))
PT_RESUME = os.environ.get("LLM_PT_RESUME") or None
pretrain_history = train_lm(
    model, pretrain_loader, pretrain_val_loader, steps=PT_STEPS,
    lr=3e-4, grad_accum=PT_GRAD_ACCUM, stage="pretrain", resume_from=PT_RESUME,
)
assert pretrain_history and pretrain_history[-1]["step"] >= PT_STEPS
assert all(math.isfinite(row["loss"]) for row in pretrain_history)
print("成功更新：", pretrain_history[-1]["step"], "有效 token：", pretrain_history[-1]["total_tokens"])
""")

code(r"""
def plot_training_history(history, title):
    # 训练曲线是随机小批损失；验证曲线是固定验证子集的加权平均。
    plt.figure(figsize=(7, 3))
    plt.plot([row["step"] for row in history], [row["loss"] for row in history], label="train")
    checked = [row for row in history if "val_loss" in row]
    plt.plot([row["step"] for row in checked], [row["val_loss"] for row in checked],
             "o-", label="validation")
    plt.xlabel("optimizer update")
    plt.ylabel("cross entropy / supervised token")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

plot_training_history(pretrain_history, "Short pretraining")
""")

code(r"""
# 只检查保存内容，不覆盖当前模型或继续训练。
pretrain_file = WORK_DIR / "pretrain_last.pt"
checkpoint_preview = torch.load(pretrain_file, map_location="cpu", weights_only=True)
assert checkpoint_preview["step"] == pretrain_history[-1]["step"]
assert checkpoint_preview["tokenizer_sha256"] == checkpoint_metadata(model)["tokenizer_sha256"]
print("checkpoint：", pretrain_file.resolve())
print("保存的字段：", sorted(checkpoint_preview))
print("数据摘要：", checkpoint_preview["data_summary"])
del checkpoint_preview  # full 权重很大，检查之后及时释放这份 CPU 副本。
""")

md(r"""
**本章练习与观察**

- 把 toy 窗口 `seq_len` 改成 `2`，手写输入和目标后再执行自检。
- 固定同一批 `[B,T]` 数据，对比一个大批与两个微批的累计梯度；关闭 dropout，并允许浮点误差。SFT 中长度不同的批次更能展示 token 加权的重要性。
- 画出 `warmup_steps=1` 和 `warmup_steps=5` 的学习率曲线，解释峰值位置。
- 在已有 checkpoint 后设置 `LLM_PT_RESUME` 与更大的 `LLM_PT_STEPS`，确认恢复从已保存的成功更新数接续，并理解为什么未保存的迭代游标会改变采样顺序。
- 正式训练要在独立验证集上比较更长训练与多个设置；本章的演示曲线不能支持模型语言能力的结论。
""")

md(r"""
## 8. SFT：教预训练模型按对话格式回答

预训练预测文本流中的所有 token；监督微调（SFT）给出问题和示范回答，主要优化回答位置。参数还是同一组 `model`，损失仍然是 next-token 交叉熵。

这一章先看对话格式和标签掩码，再加载示范数据、短训练、对比留出题。默认算术样本是教学用小数据，20 步不能证明数学推理能力或广泛的指令遵循。正式入口是本地 JSONL 数据文件。
""")

md(r"""
### 8.1 prompt、回答和 EOS 应怎样排在一起？

单轮格式：`[BOS, USER, 问题..., EOS, ASSISTANT, 回答..., EOS]`。

推理 prompt 停在 `ASSISTANT` 标记后；训练标签只保留回答 token 和回答 `EOS`。回答首 token 的目标位于 `ASSISTANT` 的输出位置，所以首 token 也必须受监督。`encode_sft` 返回**尚未移位**的等长 `ids, labels`；移位只在 `SFTDataset` 中做一次。

多轮格式重复用户/助手段；可以在开头加 `SYSTEM`。每一轮 assistant 内容都监督，system、user 和角色标记全部设成 `-100`。
""")

code(r"""
def encode_prompt(question):
    # 没有 system 消息的单轮 prompt，与下面的训练格式前缀完全一致。
    return [BOS_ID, USER_ID] + tokenizer.encode(str(question)) + [EOS_ID, ASSISTANT_ID]


def encode_sft(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空列表")
    role_ids = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}
    ids, labels = [BOS_ID], [-100]
    previous_role = None
    for index, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role not in role_ids or not isinstance(content, str):
            raise ValueError("每条消息需要合法 role 和字符串 content")
        allowed = ((role == "system" and index == 0) or
                   (role == "user" and previous_role in (None, "system", "assistant")) or
                   (role == "assistant" and previous_role == "user"))
        if not allowed:
            raise ValueError("对话必须为可选 system、user/assistant 交替，并以 assistant 结束")
        content_ids = tokenizer.encode(content)
        segment = [role_ids[role]] + content_ids + [EOS_ID]
        ids.extend(segment)
        # assistant 角色标记本身不监督，回答与终止符都监督。
        labels.extend([-100] + content_ids + [EOS_ID] if role == "assistant" else [-100] * len(segment))
        previous_role = role
    if previous_role != "assistant":
        raise ValueError("SFT 样本必须包含回答并以 assistant 结束")
    return ids, labels
""")

code(r"""
mask_demo_messages = [{"role": "user", "content": "1 + 2 = ?"},
                      {"role": "assistant", "content": "3"}]
mask_ids, mask_labels = encode_sft(mask_demo_messages)
prompt_ids = encode_prompt(mask_demo_messages[0]["content"])
assert mask_ids[:len(prompt_ids)] == prompt_ids
assert all(value == -100 for value in mask_labels[:len(prompt_ids)])
assert mask_labels[-1] == EOS_ID
assert sum(value != -100 for value in mask_labels) == len(tokenizer.encode("3")) + 1
first_answer_position = len(prompt_ids)
assert mask_ids[first_answer_position - 1] == ASSISTANT_ID
print("目标位置：", [(i, tokenizer.decode([token]), label)
                    for i, (token, label) in enumerate(zip(mask_ids, mask_labels))][-12:])
print("首回答 token 由输入位置", first_answer_position - 1, "预测；EOS 也受监督")

# 多轮自检：两个 assistant 回答的 EOS 都参加损失。
multi_ids, multi_labels = encode_sft(mask_demo_messages + [
    {"role": "user", "content": "2 + 2 = ?"}, {"role": "assistant", "content": "4"}])
assert sum(label == EOS_ID for label in multi_labels) == 2
""")

md(r"""
### 8.2 变长样本如何组成 `[B,T]`？

`SFTDataset` 把每条对话变成 `x=ids[:-1]`、`y=labels[1:]`；collate 在右边补齐 `x` 为 `PAD_ID`、`y` 为 `-100`。因果注意力中，右侧 padding 不会被左侧有效 token 看见，补齐位置也不进入损失。

本教程不默默截断过长样本，以免把回答或 EOS 切掉；它们会被跳过并报告数量。正式数据应先统计长度，必要时按完整轮次切分，或使用合适的更长上下文配置。
""")

code(r"""
class SFTDataset(Dataset):
    def __init__(self, records, max_seq_len=None):
        self.examples = []
        self.dropped = 0
        records = list(records)
        # 保存规范化内容哈希，恢复 SFT 时检查的就是这份记录，而非只检查预训练 bin。
        self.records_sha256 = hashlib.sha256(
            json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        max_seq_len = cfg.max_seq_len if max_seq_len is None else int(max_seq_len)
        for record in records:
            ids, labels = encode_sft(record["messages"])
            if len(ids) - 1 > max_seq_len:
                self.dropped += 1
                continue
            self.examples.append((torch.tensor(ids[:-1], dtype=torch.long),
                                  torch.tensor(labels[1:], dtype=torch.long)))
        if not self.examples:
            raise ValueError("没有可用 SFT 样本：请检查长度和消息格式")
        print(f"SFTDataset：保留 {len(self.examples)} 条，跳过过长样本 {self.dropped} 条")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def sft_collate(batch):
    length = max(len(inputs) for inputs, _ in batch)
    inputs = torch.full((len(batch), length), PAD_ID, dtype=torch.long)
    targets = torch.full((len(batch), length), -100, dtype=torch.long)
    for row, (item_inputs, item_targets) in enumerate(batch):
        inputs[row, :len(item_inputs)] = item_inputs
        targets[row, :len(item_targets)] = item_targets
    return inputs, targets

# 可观察自检：仅短样本的尾部被补齐，且补齐目标不参加损失。
collate_demo = SFTDataset([{"messages": mask_demo_messages}, {"messages": [
    {"role": "user", "content": "1 + 2 = ? 请写整数"}, {"role": "assistant", "content": "3"}]}])
demo_x, demo_y = sft_collate([collate_demo[0], collate_demo[1]])
first_length = len(collate_demo[0][0])
assert torch.all(demo_x[0, first_length:] == PAD_ID)
assert torch.all(demo_y[0, first_length:] == -100)
assert demo_y[0, first_answer_position - 1] == mask_ids[first_answer_position]
print("补齐后形状：", tuple(demo_x.shape), "每行有效目标：", (demo_y != -100).sum(dim=1).tolist())
""")

code(r"""
def load_sft_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "messages" not in record:
                    raise ValueError("需要含 messages 的 JSON 对象")
                encode_sft(record["messages"])  # 提前验证格式并给出准确行号。
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                raise ValueError(f"{path} 第 {line_number} 行错误：{error}") from error
            records.append(record)
    if not records:
        raise ValueError(f"{path} 不含任何有效记录")
    return records


def make_sft_loaders(train_records, val_records, batch_size=None):
    batch_size = (2 if PROFILE == "smoke" else 1) if batch_size is None else int(batch_size)
    train_dataset = SFTDataset(train_records)
    val_dataset = SFTDataset(val_records)
    generator = torch.Generator().manual_seed(2027)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
        collate_fn=sft_collate, num_workers=0, pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sft_collate,
        num_workers=0, pin_memory=DEVICE.type == "cuda",
    )
    return train_loader, val_loader
""")

md(r"""
### 8.3 教学数据：SFT 训练题与本阶段留出题分开

遍历 `a,b ∈ {0,...,9}` 可以得到 100 道加法题。预先选出 6 对作为 SFT 阶段的留出题，其他 94 对用于教学训练；验证损失和生成都在这些题上观察。这里的划分只保证它们不在 SFT 训练记录中；是否曾出现在预训练语料里，要单独审查前章的数据。它们共享相同模板，因此仍然是很窄的检验，不能据此推断对新任务的泛化。

做纯粹的记忆能力练习时，可以把训练集缩到 1～4 条并重复训练更多次，观察是否能拟合这些示范；这只能证明能拟合训练例。
""")

code(r"""
def arithmetic_question(a, b):
    return f"请只给出整数结果：{a} + {b} 等于多少？"


def arithmetic_record(a, b):
    return {"messages": [{"role": "user", "content": arithmetic_question(a, b)},
                         {"role": "assistant", "content": str(a + b)}]}

heldout_pairs = {(2, 7), (4, 6), (8, 3), (9, 9), (0, 5), (6, 1)}
demo_sft_records = [arithmetic_record(a, b) for a in range(10) for b in range(10)
                    if (a, b) not in heldout_pairs]
demo_sft_val_records = [arithmetic_record(a, b) for a, b in sorted(heldout_pairs)]
assert len(demo_sft_records) == 94 and len(demo_sft_val_records) == 6
train_questions = {record["messages"][0]["content"] for record in demo_sft_records}
val_questions = {record["messages"][0]["content"] for record in demo_sft_val_records}
assert train_questions.isdisjoint(val_questions)
print("教学训练题：", len(train_questions), "留出题：", sorted(val_questions))
""")

md(r"""
### 8.4 正式本地 JSONL 入口

设置 `LLM_SFT_TRAIN_JSONL` 和 `LLM_SFT_VAL_JSONL`，两个文件必须分别准备；每行一个记录。例如：

也可以直接把两个文件放在默认的 `data/sft_train.jsonl` 和 `data/sft_val.jsonl`；代码会自动检测。环境变量可指定其他路径，且优先于默认位置。

```json
{"messages":[{"role":"user","content":"请解释梯度累计"},{"role":"assistant","content":"多个小批先累计梯度，再执行一次参数更新。"}]}
```

支持开头的 system 与多轮 user/assistant 对话；训练记录应以 assistant 结束。正式数据需要在去重、来源与权限确认后划分，避免同一对话或近重复同时进入训练与验证。只提供一个文件时会报错，不从训练记录中临时抽几条充当独立验证。

为保持无下载的 smoke 路径，环境变量与默认本地文件均未提供时使用上述算术教学集。full 模式也可以执行这个机制演示；它不会把算术模板变成正式 SFT 语料。
""")

code(r"""
sft_train_path = os.environ.get("LLM_SFT_TRAIN_JSONL") or ("data/sft_train.jsonl" if Path("data/sft_train.jsonl").exists() else None)
sft_val_path = os.environ.get("LLM_SFT_VAL_JSONL") or ("data/sft_val.jsonl" if Path("data/sft_val.jsonl").exists() else None)
if bool(sft_train_path) != bool(sft_val_path):
    raise ValueError("请同时设置 LLM_SFT_TRAIN_JSONL 和 LLM_SFT_VAL_JSONL")
if sft_train_path:
    sft_train_records = load_sft_jsonl(sft_train_path)
    sft_val_records = load_sft_jsonl(sft_val_path)
    print("SFT 来源：本地 JSONL")
else:
    sft_train_records, sft_val_records = demo_sft_records, demo_sft_val_records
    print("SFT 来源：算术教学示例")
sft_train_loader, sft_val_loader = make_sft_loaders(sft_train_records, sft_val_records)
sft_x, sft_y = next(iter(sft_train_loader))
assert sft_x.shape == sft_y.shape and torch.any(sft_y != -100)
print("SFT 训练小批：", tuple(sft_x.shape), "有效回答 token：", int((sft_y != -100).sum()))
""")

md(r"""
### 8.5 先记录微调前的生成，再训练

训练损失回答的是“示范答案的概率是否提高”；生成还会受解码方法影响。这里固定使用贪心解码（`temperature=0`），并截取 prompt 之后、首个 EOS 之前的内容。所有留出题保持同样的推理格式。

在本地正式 JSONL 模式下，只用验证损失比较自己的数据；下列算术输出始终只是教学用 probe，不属于任意正式数据集的评估。
""")

code(r"""
def answer_question(question, max_new_tokens=12):
    prompt = encode_prompt(question)
    generated = generate(model, prompt, max_new_tokens=max_new_tokens,
                         temperature=0.0, use_cache=True)
    answer_ids = generated[len(prompt):]
    if EOS_ID in answer_ids:
        answer_ids = answer_ids[:answer_ids.index(EOS_ID)]
    return tokenizer.decode(answer_ids).strip()

sft_probe_pairs = sorted(heldout_pairs)[:3]
sft_before_answers = {pair: answer_question(arithmetic_question(*pair)) for pair in sft_probe_pairs}
sft_before_validation = evaluate_lm(model, sft_val_loader, max_batches=8 if PROFILE == "smoke" else 20)
print("SFT 前验证：", sft_before_validation)
for pair, answer in sft_before_answers.items():
    print(arithmetic_question(*pair), "→", repr(answer), "期望：", sum(pair))
""")

md(r"""
### 8.6 复用同一训练循环，只换数据和峰值学习率

SFT 的有效 token 仅来自回答，短答案尤其需要正确累计权重。本节建立新的 AdamW 状态，因为我们开始新的训练阶段；预训练模型参数继续作为起点。`LLM_SFT_RESUME` 用于恢复同一 SFT 阶段的检查点。

默认 smoke 和 full 都做 `20` 次短更新，可用 `LLM_SFT_STEPS` 改变。正式微调通常需另行设定数据量、轮次与验证频率，较低学习率只是这里的起点。
""")

code(r"""
SFT_STEPS = int(os.environ.get("LLM_SFT_STEPS", "20"))
SFT_GRAD_ACCUM = int(os.environ.get("LLM_SFT_GRAD_ACCUM", "2" if PROFILE == "smoke" else "16"))
sft_history = train_lm(
    model, sft_train_loader, sft_val_loader, steps=SFT_STEPS,
    lr=2e-4, grad_accum=SFT_GRAD_ACCUM, stage="sft",
    resume_from=os.environ.get("LLM_SFT_RESUME") or None,
)
assert sft_history and sft_history[-1]["step"] >= SFT_STEPS
assert all(math.isfinite(row["loss"]) for row in sft_history)
plot_training_history(sft_history, "Short SFT: answer-only loss")
""")

code(r"""
sft_after_validation = evaluate_lm(model, sft_val_loader, max_batches=8 if PROFILE == "smoke" else 20)
sft_after_answers = {pair: answer_question(arithmetic_question(*pair)) for pair in sft_probe_pairs}
print("SFT 验证损失：", sft_before_validation["loss"], "→", sft_after_validation["loss"])
for pair in sft_probe_pairs:
    print(f"{pair[0]}+{pair[1]}：前={sft_before_answers[pair]!r}，"
          f"后={sft_after_answers[pair]!r}，期望={sum(pair)}")
if not sft_train_path:
    # 严格字符串比较，不用解析器修补输出后再声称回答正确。
    heldout_answers = {pair: answer_question(arithmetic_question(*pair)) for pair in sorted(heldout_pairs)}
    exact_matches = sum(answer == str(sum(pair)) for pair, answer in heldout_answers.items())
    print(f"教学留出题严格匹配：{exact_matches}/{len(heldout_answers)}；样本太少，不代表广泛能力")
print("SFT checkpoint：", (WORK_DIR / "sft_last.pt").resolve())
""")

md(r"""
**本章练习与下一步**

- 把一条对话写成两轮，逐 token 检查哪些标签是 `-100`，解释两个回答 EOS 为什么都受监督。
- 故意把 padding 目标设成 `PAD_ID`，观察损失计算目标数的变化，再改回 `-100`；这展示了掩码如何改变训练任务。
- 只用 4 条示范训练较多步，记录训练例拟合和留出题结果。训练例拟合成功不等于获得通用加法能力。
- 接入自己的本地训练/验证 JSONL，报告保留与跳过数量、验证有效 token 数及曲线；不要只展示几个挑选出的正确回答。

此时 `model` 保留 SFT 后的参数，`sft_history` 保留更新记录，后续章节可以继续用它作为强化学习的初始策略。正式实验应保存 SFT 检查点作为可复用起点，并按下一章的规则另建参考模型。
""")
