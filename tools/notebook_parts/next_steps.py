"""完整训练路线、复习与资料的教学 cell。"""
from textwrap import dedent

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "source": dedent(source).strip()})


def code(source):
    CELLS.append({"cell_type": "code", "source": dedent(source).strip()})


md(r"""
### 10.5 验证集与测试集：不同用途的两次检查

训练时的 `val_tokens` 用于观察和调整设置；`test_tokens` 留到设置确定后作最终报告，不能看过结果后不断针对它调参还称为独立测试。下面对当前最终状态的模型计算一小份固定测试文本的 token 加权 loss 和 perplexity。

这不是纯预训练 checkpoint 的评估，因为模型已经历 SFT 和 RL 尝试。正式实验应分别重载各阶段的权重，用同样的测试文本比较。默认演示语料高度重复，所以这个数只说明整个评估流程能工作；真实语言能力仍需要独立、足量的文本与任务测试。
""")

code(r"""
# 不用随机 sampler；每次都评同一组均匀分布的文本窗口。
test_dataset = TokenWindowDataset(test_tokens, cfg.max_seq_len)
test_starts = np.linspace(0,len(test_dataset)-1,min(32,len(test_dataset)),dtype=np.int64).tolist()
test_loader = DataLoader(Subset(test_dataset,test_starts),batch_size=1,num_workers=0)
final_text_test = evaluate_lm(model,test_loader)
print("最终状态的固定测试文本子集:", final_text_test)
print("这是测试子集的指标；smoke模板语料不支持通用语言能力结论。")
""")

md(r"""
## 11. 从教学小实验切换到真正的 0.1B 训练

你已经用同一套代码串起各阶段。现在把目标从“正确运行”变成“从真实数据学习”。**0.1B 指参数量，不是训练 token 数**。只切换配置而不增加数据，模型可能只是更容易记住小语料。

依次执行下面的步骤，每次只改变一个因素：

1. **准备真实文本。** 把 txt/jsonl 放在 `data/raw/`；优先限定语言与领域，检查字符编码、质量、重复和 train/val/test 的同源污染。JSONL 每行是 `{"text":"完整文档"}`。
2. **重启内核，改控制面板的 `PROFILE` 为 `full`。** 从头运行到模型定义，确认词表 8192、参数 98,324,736。不要把旧 smoke checkpoint 用在新模型上。
3. **先做 10～20 次 optimizer 更新。** 观察有限的 loss/梯度、显存、tokens/s、验证是否运行、checkpoint 是否可恢复，再考虑长训练。必要时用环境变量 `LLM_PT_STEPS` 设短步数；它是总 optimizer 更新目标。
4. **根据 token 预算设更新次数。** 保持模型配置、分词器、数据与有效 batch 固定，再增加训练步数。先完成一个足够小的里程碑，评估后继续。
5. **完成基座模型验证再做 SFT。** 自己的对话放在 `data/sft_train.jsonl`、`data/sft_val.jsonl`，每行 `{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"回答"}]}`。第 8 章会读取这些文件，或使用合成教学数据。真实 SFT 要让格式、语言、答案质量与目标场景一致。
6. **只有任务已有成功样本时才认真尝试 RL。** 如果模型从不答对题，可验证奖励全为 0，RL 无从比较优劣；先改善数据/预训练/SFT。演示代码跳过这些组，并显示原因。

8 GiB 显存的起始配置：context 256、micro batch 1、梯度累计、SDPA、AMP、激活重计算。具体上限取决于软件与数据；出现 OOM 时先减 context/micro batch、清理不再使用的模型引用并重启内核，不要反复构建多个完整模型。
""")

md(r"""
### 11.1 把“训练多久”变成可计算的预算

设 micro batch 为 $b$，梯度累计次数为 $a$，每条序列长度为 $T$，GPU 数量为 $g$：

$$\text{有效 batch}=bag,\qquad \text{预训练 token/更新}=bagT.$$

本教程只有一张 GPU，$g=1$。SFT 的有效监督 token 因回答长度与 padding 而变化，不能直接套固定 $T$ 的公式。

例如 $b=1,a=16,T=256$，每次预训练更新 4096 个 token；重复小数据也会计入这个数，**处理次数并不等于独立数据量**。计算预算时同时记下唯一语料 token 数、重复次数和处理 token 总数。

可先探索 10M、100M、1B 等逐级预算，并观察保留文本的验证损失和生成质量。研究中的约 `20×参数量` token 比例可作为一个历史计算参考，不能保证本模型、本中文词表或本数据达到某种质量；不同训练目标的最优预算也会变化。来源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)。
""")

code(r"""
# 这里只算预算，不触发训练。把这些数改成你的实际配置。
budget_micro_batch, budget_accum, budget_context = 1, 16, 256
tokens_per_update = budget_micro_batch * budget_accum * budget_context
for target_tokens in [10_000_000, 100_000_000, 1_000_000_000, 2_000_000_000]:
    required_updates = math.ceil(target_tokens/tokens_per_update)
    print(f"{target_tokens:,} token -> {required_updates:,} optimizer 更新（每次 {tokens_per_update:,} token）")

# 示例吞吐是假设值；请换成自己训练日志中的实测 tokens/s。
assumed_tokens_per_second = 1000
print("假设每秒1000 token，100M token约需小时:", round(100_000_000/assumed_tokens_per_second/3600,1))
print("该时长只是算术示例，不是本机测速或训练承诺。")
""")

md(r"""
### 11.2 显存与吞吐实验：确认完整模型能训练一步

下面是一个**主动开启**的小探针，默认关闭。它构建独立的 98.3M 模型，在随机 token 上做一次 AdamW 更新，观察 AMP 和激活重计算能否工作；不是语言能力评估，也不会替换你训练的模型。

若当前正在持有完整模型、优化器、reference 等，先重启内核并只运行到模型定义，或使用项目终端 `python tools/validate_notebook.py --full-probe`。后者验证后会释放探针。随机 token 的 loss 大约接近 $\log V$，并无学习语言的含义。
""")

code(r"""
RUN_FULL_DEVICE_PROBE = False  # 想实际检查完整模型的一步训练时改为 True
if RUN_FULL_DEVICE_PROBE:
    if DEVICE.type != "cuda":
        raise RuntimeError("完整模型资源探针请使用 CUDA GPU；CPU 可以继续学习 smoke。")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    probe_model = TinyLLM(FULL_CFG).to(DEVICE).train()
    probe_optimizer = torch.optim.AdamW(probe_model.parameters(),lr=1e-4,foreach=False)
    probe_scaler = torch.amp.GradScaler("cuda",enabled=AMP_DTYPE==torch.float16)
    probe_ids = torch.randint(0,FULL_CFG.vocab_size,(1,257),device=DEVICE)
    torch.cuda.synchronize()
    probe_started = time.perf_counter()
    with amp_context():
        probe_logits, _ = probe_model(probe_ids[:,:-1])
        probe_loss = F.cross_entropy(probe_logits.reshape(-1,FULL_CFG.vocab_size),probe_ids[:,1:].reshape(-1))
    probe_scaler.scale(probe_loss).backward()
    probe_scaler.unscale_(probe_optimizer)
    probe_norm = torch.nn.utils.clip_grad_norm_(probe_model.parameters(),1.0)
    assert torch.isfinite(probe_loss) and torch.isfinite(probe_norm)
    probe_scaler.step(probe_optimizer)
    probe_scaler.update()
    torch.cuda.synchronize()
    print("参数:", sum(p.numel() for p in probe_model.parameters()))
    print("单步秒数:", round(time.perf_counter()-probe_started,3), "峰值已分配GiB:", round(torch.cuda.max_memory_allocated()/2**30,3))
    del probe_model, probe_optimizer, probe_scaler, probe_ids, probe_logits, probe_loss, probe_norm
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("完整模型资源探针关闭；需要时手动开启。")
""")

md(r"""
### 11.3 复习：把系统讲清楚，才算真正掌握

先不看代码，尝试用自己的话回答：

1. 一个 token ID 为什么不是一个固定语义向量？Tokenizer 与 Embedding 分别学什么？
2. Q/K/V 怎样从 `[B,T,d]` 变成多头 shape？为什么注意力分数除以 $\sqrt{d_h}$？
3. 哪个测试能发现偷看未来？为什么训练和生成的计算方式不同？
4. $98,324,736$ 中每一项来自哪一块？增大 context 为什么不增加参数？
5. 已 shift 的标签为什么不能在 loss 里再次 shift？SFT 第一段回答的第一个 token 由哪个位置预测？
6. 梯度累计、梯度裁剪和 AMP scaler 分别解决什么？更新次数与 micro batch 次数怎样区分？
7. 预训练与 SFT 的模型结构是否改变？只有损失下降，能说明模型能聊天吗？
8. 同一 prompt 的所有 reward 一样，为什么组内 advantage 没有信息？
9. RL 的 `old_logp` 为什么不能跟着当前参数求梯度？top-p 为什么影响概率比的解释？
10. KV Cache 应保存什么？缓存时 RoPE 位置偏移和因果 mask 为何容易出错？

然后完成三个小项目：

- **理解项目：** 在小数据上去掉 RoPE、减少层数、取消权重共享，一次改一项，比较参数量与验证损失。用相同数据、种子和 token 预算控制变量。
- **训练项目：** 用一个限定主题的真实语料训练基座模型，保留三组从未参与选模型的提示，逐级记录续写与验证损失。
- **对话项目：** 写 50 条高质量、互不重复的指令训练数据和一组独立测试问题；比较 SFT 前后回答格式，再加入可验证任务。这个规模用于学习数据作用，仍不能证明通用能力。

保存一份实验表：配置、数据摘要、token 预算、loss、评估题集合、答案样例。避免只保存“最漂亮的一条回答”。
""")

md(r"""
## 12. 继续阅读：用原始资料对照已经写过的代码

| 已学概念 | 原始资料/官方文档 | 对照本教材 |
|---|---|---|
| BPE 训练与持久化 | [Tokenizers Quicktour](https://huggingface.co/docs/tokenizers/quicktour) | 自己的 train 语料学习 merge，而非加载模型权重 |
| Attention mask 与 SDPA | [PyTorch 2.9 SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) | 手写路径、布尔 mask、缓存非方阵 |
| AMP 与 GradScaler | [PyTorch 2.9 AMP](https://docs.pytorch.org/docs/2.9/amp.html) | 先 unscale 再裁剪，FP16 与 BF16 的区别 |
| 旋转位置编码 | [RoFormer](https://arxiv.org/abs/2104.09864) | RoPE 的坐标配对与位置偏移 |
| 参数/数据/计算预算 | [Chinchilla](https://arxiv.org/abs/2203.15556) | token 预算只是参考，实际质量需要验证 |
| 组相对策略优化 | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 本教材是简化教学实现，之后再读原始 GRPO |
| 人类偏好强化学习 | [InstructGPT](https://arxiv.org/abs/2203.02155) | 更完整的 RLHF：偏好标注、奖励模型与策略优化 |
| 直接偏好优化 | [DPO](https://arxiv.org/abs/2305.18290) | 偏好学习的另一条路线，不能把它等同于在线 RL rollout |

教材代码优先保证机制可见、可验证。后续工程扩展可以加入分布式训练、更长 context、更丰富评估、推理服务和批量 KV Cache；每次扩展前保留本教材的因果性、mask、缓存一致性与 checkpoint 检查。
""")
