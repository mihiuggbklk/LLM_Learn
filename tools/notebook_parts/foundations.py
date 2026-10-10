"""教学 Notebook 的前半部分；这里只保存 cell 源码，学习者无需导入此文件。"""
from textwrap import dedent

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "source": dedent(source).strip()})


def code(source):
    CELLS.append({"cell_type": "code", "source": dedent(source).strip()})


md(r"""
# 从零设计并实现自己的 0.1B LLM
## 一本以理解为中心的 Python / PyTorch 实验教材

你将自己实现一个 **Decoder-only Transformer**，从随机参数开始，依次完成数据与分词器、模型、预训练、SFT、RLVR，最后实现有 KV Cache 的多轮对话。模型不加载任何现成的预训练权重。

**目标模型：98,324,736 个可训练参数，约 0.1B。** `B` 是十亿，`0.1B` 是一亿。它是适合学习的小语言模型；完整训练需要足够的数据和计算，仅运行本教材的短实验不会得到 ChatGPT 的知识与对话质量。

**使用方法：** 选择 Python 内核，从上往下执行。先读文字，预测输出，再运行代码核对。代码中的中文注释解释关键操作；每章都有检查点和小练习。不要一次跳过所有解释直接运行到底。

### 两条运行路线

| 路线 | 模型与数据 | 目的 |
|---|---|---|
| `smoke`，默认 | 相同架构的小模型、自带离线小语料、短训练 | 观察张量、梯度和整个训练流程 |
| `full` | 98.3M 参数、你准备的真实语料、自训 8192 词表 | 在理解后进行 0.1B 模型的实际训练 |

两个配置调用同一套网络代码。小模型用于快速验证思路，完整配置决定真正的参数规模。默认短训练与生成输出都是**教学实验**，不能用来宣称模型已经具备可靠能力。
""")

md(r"""
## 学习地图与完成标准

| 章 | 你要回答的问题 | 完成这一章的证据 |
|---|---|---|
| 0 环境 | Python 与 GPU 如何协作？ | 正确的内核、设备和精度输出 |
| 1 概率与张量 | “预测下一个词”具体预测什么？ | 手算与 PyTorch 交叉熵一致 |
| 2 Tokenizer | 文字怎么变成整数？BPE 合并什么？ | 编码/解码往返一致 |
| 3 数据 | 怎样避免验证集泄漏？ | 清洗去重后分割、token 数与 fingerprint |
| 4 参数设计 | 怎样得到约一亿参数？ | 参数公式与实际统计一致 |
| 5 模型 | Attention、RoPE、SwiGLU 各做什么？ | 逐层 shape、因果性和梯度检查 |
| 6 生成 | 怎么从一次前向变成续写？ | 缓存与全量前向 logits 一致 |
| 7 预训练 | 数据怎样改变参数？ | 短训练、验证损失、保存与恢复 |
| 8 SFT | 为什么只监督 assistant 回答？ | prompt/padding 不计 loss，EOS 计 loss |
| 9 RLVR | 奖励怎样改变回答的概率？ | advantage、policy loss 和真实 rollout |
| 10 评估与聊天 | 怎样区分会背题与会做题？ | 保留题评估、对话与模型导出 |
| 11 走向 0.1B | 长训练需要怎样安排？ | 实际 token 预算、续训方法与学习练习 |

建议分几次学习：第一次到参数设计；第二次手写模型；第三次预训练；第四次 SFT；第五次 RLVR 与评估。你需要会 Python 函数、类和列表；尚不熟悉张量与自动求导时先做第 1 章。
""")

md(r"""
## 0. 准备环境：先让内核使用正确的 Python

本教程已在本地 **PyTorch 2.9.1 + CUDA 13.0、tokenizers 0.22.1** 环境验证。教程依赖安装到你选择的 Notebook 内核；不要因为系统终端里能导入 `torch` 就假设当前内核也可以。

在项目终端执行 `python -m pip install -r requirements.txt`，然后 `python -m jupyter lab`，打开本文件。GPU 版 PyTorch 的安装命令与驱动、平台有关，使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择；不要为了本教程重装已经能运行 CUDA 的 PyTorch。

下面这格是本教材的**控制面板**。默认不下载数据、不开启长时间训练。第一次保持 `PROFILE = "smoke"`。准备完整训练时，按第 11 章先放好真实数据，再改成 `full` 并重启内核，从头运行。
""")

code(r"""
# 基础工具：所有真正的模型、训练与采样代码都将在后续 cell 中展开。
import os, sys, json, math, time, random, hashlib, re, gc, unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from collections import Counter
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt

# Jupyter 中把图画在 cell 输出区；脚本验证时不创建外部图窗。
if "get_ipython" in globals() and get_ipython() is not None:
    get_ipython().run_line_magic("matplotlib", "inline")

# 可直接改成 "full"；环境变量用于测试脚本或终端启动，普通学习无需设置。
PROFILE = os.environ.get("LLM_PROFILE", "smoke")
assert PROFILE in {"smoke", "full"}
SEED = 42
WORK_DIR = Path(os.environ.get("LLM_ARTIFACT_DIR", f"artifacts/{PROFILE}"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("data/raw")  # full 路线：放自己的 txt/jsonl 语料

# 固定随机种子便于比较实验；CUDA 某些算子仍可能有细小数值差异。
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEVICE.type == "cuda" else torch.float32

def amp_context():
    # CPU 保持 FP32；CUDA 在线性运算中使用更省显存的精度。
    return torch.autocast("cuda", dtype=AMP_DTYPE) if DEVICE.type == "cuda" else nullcontext()

print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "设备:", DEVICE, "精度:", AMP_DTYPE)
if DEVICE.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name, "显存 GiB:", round(props.total_memory / 2**30, 2))
print("路线:", PROFILE, "产物目录:", WORK_DIR.resolve())
""")

md(r"""
**观察：** 本机应显示 RTX 4060 Laptop GPU、约 8 GiB 显存和 `cuda`。CPU 也能完成小实验，只是更慢。

**检查点：** 你能解释磁盘上的模型文件、CPU 内存中的参数和 GPU 显存中的张量这三者的区别吗？后面 `x.to(DEVICE)` 是一次设备移动，不是“把 Python 代码搬到 GPU”。
""")

md(r"""
## 1. 先认识模型输出：整数、logits、概率与 loss

假设词表只有三个词：`我、爱、学习`。输入“我 爱”，模型对下一个词输出三个实数 **logits**。Softmax 将它们变成总和为 1 的概率。

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad \mathcal L=-\log p_{\text{正确词}}.$$

logits 没有“必须在 0 到 1”这一限制。正确词的概率越低，交叉熵越高。这里先对一个位置手算，再对照 PyTorch。预测：如果把正确词的 logit 调高，loss 会变大还是变小？
""")

code(r"""
# [B=1, V=3]：只有一个预测位置、三个候选词。
toy_logits = torch.tensor([[0.2, -0.3, 1.1]], requires_grad=True)
toy_target = torch.tensor([2])  # 正确答案为“学习”
toy_probs = toy_logits.softmax(dim=-1)
manual_loss = -toy_probs[0, toy_target.item()].log()
pytorch_loss = F.cross_entropy(toy_logits, toy_target)
torch.testing.assert_close(manual_loss, pytorch_loss)
pytorch_loss.backward()  # 自动求导：告诉我们每个 logit 应怎样调整
print("概率:", toy_probs.detach().tolist(), "loss:", round(pytorch_loss.item(), 4))
print("logits 的梯度:", toy_logits.grad.tolist())
assert toy_logits.grad[0, 2] < 0  # 梯度下降会提高正确词的 logit
""")

md(r"""
### 1.1 一次训练同时预测很多位置

输入 `[我, 爱, 学习]`，目标为 `[爱, 学习, EOS]`。模型在第 0 个位置预测第 1 个 token，最后一个位置预测结束符。**输入和目标相差一位**，这是整个教程最容易写错的地方。

下面的网络只看当前 token，是一个很弱的 bigram 模型。先拿它理解 Embedding、batch、反向传播，第 5 章再学习怎样利用更长的上下文。
""")

code(r"""
# [B,T] 的整数 -> [B,T,V] 的 logits。Embedding 是可学习的查表，不是分词器。
tiny_bigram = nn.Embedding(4, 4)
bigram_x = torch.tensor([[0, 1, 2]])
bigram_y = torch.tensor([[1, 2, 3]])
bigram_opt = torch.optim.AdamW(tiny_bigram.parameters(), lr=0.15)
bigram_losses = []
for _ in range(30):
    bigram_opt.zero_grad(set_to_none=True)
    logits = tiny_bigram(bigram_x)  # [1,3,4]
    loss = F.cross_entropy(logits.reshape(-1, 4), bigram_y.reshape(-1))
    loss.backward()
    bigram_opt.step()
    bigram_losses.append(loss.item())
print("shape:", logits.shape, "loss:", round(bigram_losses[0], 3), "->", round(bigram_losses[-1], 3))
assert bigram_losses[-1] < bigram_losses[0]
""")

md(r"""
**检查点：** `reshape(-1, V)` 将 batch 与时间合并，每一行是一道“下一 token”分类题。标签必须是 `torch.long`，不是 one-hot 向量。

**练习：** 把训练序列改成 `[0,1,0,2]`。同一个 `0` 有时接 `1`，有时接 `2`，bigram 为什么无法可靠地区分？这个矛盾解释了我们为什么需要能读取上下文的 Transformer。
""")

md(r"""
## 2. Tokenizer：先自己写一个最小 BPE

字符分词简单，但中文、英文、数字需要的序列长度可能差很多。BPE 从很小的符号开始，反复合并最常见的相邻符号，让常见片段成为一个 token。

例子：`low low lower` 中，先统计相邻字符对，可能把 `l,o` 合成 `lo`，再把 `lo,w` 合成 `low`。以下代码只演示**合并规则怎样被学出来**；它不是用于大型语料的高性能分词器，也没有完成特殊 token、字节覆盖和文件持久化。
""")

code(r"""
def merge_pair(sequence, pair, merged):
    # 从左到右合并不重叠的相邻 pair，例如 a a a -> aa a。
    result, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            result.append(merged)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return result

def learn_toy_bpe(text, merges=5):
    # 此处的符号直接用字符串，方便看清每轮合并。
    sequence, rules = list(text), []
    for _ in range(merges):
        pairs = Counter(zip(sequence, sequence[1:]))
        if not pairs:
            break
        pair, count = pairs.most_common(1)[0]
        merged = pair[0] + pair[1]
        sequence = merge_pair(sequence, pair, merged)
        rules.append((pair, merged))
        print("合并:", pair, "次数:", count, "结果:", sequence)
    return sequence, rules

toy_pieces, toy_rules = learn_toy_bpe("low low lower low", merges=4)
assert "".join(toy_pieces) == "low low lower low"
""")

md(r"""
**练习：** 换成“学习 学习 学生”，先手算第一轮。再思考：训练得到的规则在编码新文本时也必须按原顺序应用，为什么不能重新统计新文本的词频？

### 2.1 实际使用 byte-level BPE：先划分数据，再训练分词器

实际分词器用 `tokenizers` 库执行合并算法，但合并规则来自**自己的训练语料**，没有加载外部现成分词器。预先加入完整的 256 个字节符号，让未见过的字符也可以由 UTF-8 字节表示；一个中文字符因此可能对应多个 token。

分词器也是从数据学习的对象，所以先清洗、去重、划分 train/val/test，再只用 train 学规则。为了看清这个顺序，先做第 3 章的数据准备，然后回到本章的实际分词器代码。这种安排避免验证文本参与学习词表。
""")

md(r"""
## 3. 原始数据：小语料用于观察，真实语料用于学习能力

默认自带的文字和算术语料让 Notebook 离线运行。它们数量少、模板重复，验证损失只能检查训练流程，不能证明自然语言泛化。

`full` 路线读取 `data/raw/` 内的文件：`.txt` 每个非空行是一篇文档；`.jsonl` 每行形如 `{"text":"一篇完整文档……"}`。对大文档可以先切成段落；不要把题目与答案拆到两个集合。选择你有权用于训练、语言和领域合适的语料。

数据处理顺序为 **统一格式 → 去除精确重复 → 按文档哈希划分 → 学词表 → 转 token 文件**。这里只做精确去重，近似重复与同源文档应在正式训练前另外分组处理。
""")

code(r"""
def demo_documents():
    # 这些模板是自写的离线教材数据；形式重复，不能充当真实泛化基准。
    concepts = [
        "语言模型通过预测下一个符号学习文本中的规律。",
        "注意力把相关位置的信息按权重组合起来。",
        "训练集用于更新参数，验证集用于选择训练策略。",
        "梯度下降沿着损失减小的方向调整参数。",
        "Python 的列表可以保存整数，张量可以执行矩阵乘法。",
        "模型先进行预训练，再通过指令微调学习对话格式。",
        "A language model predicts the next token from previous tokens.",
        "Validation data helps us inspect generalization during training.",
    ]
    for i in range(400):
        yield f"教学笔记第{i}页。{concepts[i % len(concepts)]}今天观察第{i % 37}次实验，认真记录输入、输出与损失。"
    for a in range(10):
        for b in range(10):
            yield f"算术练习：{a} + {b} = {a+b}。先计算，再核对最终答案。"

def raw_documents(directory):
    # 流式逐行读取，避免把全部原始文本一次放入内存。
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".txt", ".jsonl"})
    if not files:
        raise FileNotFoundError(f"请先在 {directory.resolve()} 放 txt/jsonl 真实语料；第一次学习可用 smoke。")
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if path.suffix.lower() == ".jsonl":
                    record = json.loads(line)
                    if not isinstance(record.get("text"), str):
                        raise ValueError(f"{path}:{line_number} 必须有字符串 text 字段")
                    yield record["text"]
                else:
                    yield line

def clean_document(text):
    # 保留空格与换行语义；不粗暴删掉数字或标点。
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text if len(text) >= 16 else None

def prepare_splits(documents, directory):
    paths = {name: directory / f"{name}.jsonl" for name in ("train", "val", "test")}
    counts = dict.fromkeys(paths, 0)
    seen = set()  # 内存只存摘要；数千万篇时应换成 SQLite/外部排序。
    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for raw in documents:
            text = clean_document(raw)
            if text is None:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            bucket = int(digest[:8], 16) % 10
            split = "train" if bucket < 8 else ("val" if bucket == 8 else "test")
            handles[split].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    if min(counts.values()) == 0:
        raise ValueError(f"有空集合 {counts}；请补充文档，不要把验证/测试文本复制进训练集。")
    return paths, counts

split_paths, document_counts = prepare_splits(demo_documents() if PROFILE == "smoke" else raw_documents(RAW_DIR), WORK_DIR)
print("去重后的文档数:", document_counts)
""")

md(r"""
### 3.1 训练自己的分词器，并为对话预留边界

我们使用 6 个特殊符号：padding、BOS（开始）、EOS（结束）、user、assistant、system。它们和普通中文、英文一样对应一个整数 ID，但边界符不能被拆成几个片段。

`8192` 是完整配置的目标词表，不是语料很小时一定能学出的数量。真实语料不足以产生 8192 个符号时，本教材会报错并提示补充数据，避免悄悄改变 0.1B 设计。默认小实验使用较小词表。

这里采用完整字节覆盖、不过滤空格的分词方案，方便验证 `decode(encode(text)) == text`。有关 BPE 的库接口见 [Tokenizers 官方教程](https://huggingface.co/docs/tokenizers/quicktour)。
""")

code(r"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>"]

def iter_split_text(path):
    # 可以重复调用迭代器，从磁盘再次读取 train，无需保留全部字符串。
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)["text"]

class TextTokenizer:
    # 轻量包装，让后续训练代码只依赖 encode/decode/vocab_size 三个接口。
    def __init__(self, backend):
        self.backend = backend
        self.vocab_size = backend.get_vocab_size()

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode([int(x) for x in ids], skip_special_tokens=True)

target_vocab_size = 1024 if PROFILE == "smoke" else 8192
bpe = Tokenizer(BPE())
# 不自动添加前导空格；标准预切分避免把整篇文档合成一个超长 token。
bpe.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
bpe.decoder = ByteLevelDecoder()
bpe.train_from_iterator(
    iter_split_text(split_paths["train"]),
    trainer=BpeTrainer(vocab_size=target_vocab_size, min_frequency=2,
                       special_tokens=SPECIAL_TOKENS, initial_alphabet=ByteLevel.alphabet()),
)
tokenizer = TextTokenizer(bpe)
if PROFILE == "full" and tokenizer.vocab_size != 8192:
    raise ValueError(f"只学出 {tokenizer.vocab_size} 个 token；请增加真实训练语料后再训练 8192 词表。")
bpe.save(str(WORK_DIR / "tokenizer.json"))
PAD_ID, BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID, SYSTEM_ID = [bpe.token_to_id(s) for s in SPECIAL_TOKENS]
probe_text = "你好，LLM! 12 + 13 = 25 🐱\n学习 Python。"
probe_ids = tokenizer.encode(probe_text)
assert tokenizer.decode(probe_ids) == probe_text
assert all(len(tokenizer.encode(s)) == 1 for s in SPECIAL_TOKENS)
print("实际词表:", tokenizer.vocab_size, "特殊ID:", [PAD_ID,BOS_ID,EOS_ID,USER_ID,ASSISTANT_ID,SYSTEM_ID])
print("编码:", probe_ids, "\n解码:", tokenizer.decode(probe_ids))
""")

md(r"""
**检查点：** 分词器的 ID 不能在预训练后随意改变。换词表会让同一个整数代表另一个词，而 Embedding 仍保留旧含义。后续 checkpoint 会记录分词器摘要，恢复时核对。

**练习：** 比较中文句子、英文句子和连续数字的 token 数。词表扩大可能缩短序列，但也增加 Embedding 与 LM Head 的开销；第 4 章会把这项代价写成公式。
""")

md(r"""
### 3.2 从文档变成训练用的 token 流

把文档编码后接上 EOS，再顺序写入二进制文件。`memmap` 让我们按需读取 token，避免把长语料全部变成 GPU 张量。只在取一个 batch 时才转 `torch.long` 并搬到设备。

不同文档之间用 EOS 分隔。本入门实现允许文档之间做因果注意力，EOS 是边界提示；大型训练还可使用文档级注意力掩码。训练、验证、测试始终写在三个不同文件中。
""")

code(r"""
def file_sha256(path):
    # 分块哈希，不把整个 token 文件读进内存。
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_token_stream(jsonl_path, binary_path):
    count = 0
    with binary_path.open("wb") as output:
        for text in iter_split_text(jsonl_path):
            ids = tokenizer.encode(text) + [EOS_ID]
            assert max(ids) < 65536  # uint16 的范围，本教程词表小于此值
            np.asarray(ids, dtype="<u2").tofile(output)
            count += len(ids)
    return count

token_counts = {}
manifest = {"profile": PROFILE, "documents": document_counts,
            "tokenizer_sha256": file_sha256(WORK_DIR / "tokenizer.json"), "splits": {}}
for split, path in split_paths.items():
    output_path = WORK_DIR / f"{split}.bin"
    token_counts[split] = write_token_stream(path, output_path)
    manifest["splits"][split] = {"tokens": token_counts[split], "sha256": file_sha256(output_path)}
DATA_FINGERPRINT = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
manifest["fingerprint"] = DATA_FINGERPRINT
(WORK_DIR / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
train_tokens, val_tokens, test_tokens = [np.memmap(WORK_DIR / f"{s}.bin", dtype="<u2", mode="r") for s in ("train", "val", "test")]
print("token 数:", token_counts, "\nfingerprint:", DATA_FINGERPRINT[:16])
print("一段训练文本:", tokenizer.decode(train_tokens[:80]))
""")

md(r"""
## 4. 先算参数，再设计 0.1B 模型

设词表大小 $V$，隐藏维度 $d$，层数 $L$，SwiGLU 中间维度 $m$。使用无 bias 的线性层、每层两个 RMSNorm、Embedding 与输出权重共享时：

$$N = Vd + L(4d^2 + 3dm + 2d) + d.$$

- $Vd$：Embedding 与 LM Head 共享同一张参数表，因此只计一次。
- $4d^2$：Q、K、V、输出投影，四个 $d\times d$ 矩阵。
- $3dm$：SwiGLU 的两个升维矩阵和一个降维矩阵。
- $2d$：每个 block 两个 RMSNorm 的缩放向量。
- 最后的 $d$：输出前的 RMSNorm。RoPE 使用固定函数，没有可训练位置参数。

我们选 **$V=8192,d=768,L=13,m=2048$**，得到 **98,324,736**。12 个注意力头使每头维度为 64；头数本身不直接增加参数，但改变信息处理和计算布局。13 层虽然不是常见整数倍，架构上完全合法。

为什么选 $m=2048$？普通两层 FFN 若中间维度为 $4d$，参数是 $8d^2$；SwiGLU 有三张矩阵，令 $3dm=8d^2$，得到 $m=8d/3=2048$。这样在引入门控时保持类似的参数预算。
""")

code(r"""
@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 768
    n_layers: int = 13
    n_heads: int = 12
    hidden_dim: int = 2048
    max_seq_len: int = 256
    use_sdpa: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "hidden dim 必须能均分成注意力头"
        assert (self.dim // self.n_heads) % 2 == 0, "RoPE 每头维度必须为偶数"
        assert min(self.vocab_size,self.dim,self.n_layers,self.n_heads,self.hidden_dim,self.max_seq_len) > 0

def parameter_budget(config):
    V, d, L, m = config.vocab_size, config.dim, config.n_layers, config.hidden_dim
    return V*d + L*(4*d*d + 3*d*m + 2*d) + d

FULL_CFG = ModelConfig()
cfg = FULL_CFG if PROFILE == "full" else ModelConfig(
    vocab_size=tokenizer.vocab_size, dim=64, n_layers=2, n_heads=4,
    hidden_dim=176, max_seq_len=64, use_sdpa=True, gradient_checkpointing=False,
)
assert parameter_budget(FULL_CFG) == 98_324_736
print("完整目标:", f"{parameter_budget(FULL_CFG):,}", "参数")
print("本次路线:", asdict(cfg), "\n本次参数:", f"{parameter_budget(cfg):,}")
""")

md(r"""
### 4.1 一亿参数在 8 GiB 显存里占多少？

采用 FP32 参数 + AdamW 时，参数、梯度、两份优化器动量约为 **$16N$ 字节**。98.3M 参数约占 1.47 GiB。这还没算激活值、logits、CUDA 工作区、显存碎片、RL 的 reference 副本等。

所以“参数能放下”不等于“任意 batch 都能训练”。本机从 micro batch 1、context 256、AMP、激活重计算开始。梯度累计增加有效 batch，通常不增加单个 micro batch 的激活峰值。先跑短步测显存，再调整。

注意：AMP 是按算子混用精度，本教程的模型参数仍为 FP32；不要在 AMP 训练前再随意 `model.half()`。参考 [PyTorch 2.9 AMP 文档](https://docs.pytorch.org/docs/2.9/amp.html)。
""")

code(r"""
# 这是静态估算，不是承诺的实际峰值；真实峰值在第 11 章测量。
N = parameter_budget(FULL_CFG)
print("仅 FP32 参数 MiB:", round(N*4 / 2**20, 1))
print("参数+梯度+AdamW 状态 GiB:", round(N*16 / 2**30, 2))
print("batch=1,context=256 的 FP32 logits MiB:", round(1*256*8192*4 / 2**20, 1))
""")

md(r"""
**练习：** 只改词表到 16384，算参数增加多少；只把 context 改到 512，为什么参数量不变但训练显存会上升？写下答案，再继续手写网络。
""")
