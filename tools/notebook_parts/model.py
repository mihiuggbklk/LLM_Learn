"""模型和生成部分的中文教学 cell。"""
from textwrap import dedent

CELLS = []


def md(source):
    CELLS.append({"cell_type": "markdown", "source": dedent(source).strip()})


def code(source):
    CELLS.append({"cell_type": "code", "source": dedent(source).strip()})


md(r"""
## 5. 手写 Transformer：先看数据怎样流动

以下维度贯穿本章：$B$ 是 batch，$T$ 是 token 数，$d$ 是隐藏维度，$H$ 是头数，$d_h=d/H$ 是每头维度，$V$ 是词表大小。

```text
整数 ID [B,T]
  ↓ Embedding：查表
隐藏状态 [B,T,d]
  ↓ 重复 L 次 Transformer Block
  ├─ RMSNorm → Causal Attention（Q、K 上施加 RoPE）→ 加回残差
  └─ RMSNorm → SwiGLU → 加回残差
  ↓ 最后的 RMSNorm
  ↓ 共享 Embedding 的线性输出层
logits [B,T,V]
```

一个 block 改变表示，保持 `[B,T,d]` 形状。这样才能堆叠多层。不要一次复制整套模型；先实现每个部件，并在小张量上检查它。
""")

md(r"""
### 5.1 RMSNorm：稳定每个 token 的特征尺度

对一个 token 的 $d$ 维向量 $x$：

$$\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac1d\sum_j x_j^2+\epsilon}}\,g_i.$$

$g$ 是可学习的缩放向量，初值为 1。它沿**最后的特征维度**归一化，不会混合不同 token。与 LayerNorm 不同，它不减均值。平方与求均值在 FP32 做，以减少低精度数值问题，再转回输入类型。
""")

code(r"""
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        original_dtype = x.dtype
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms * self.weight.float()).to(original_dtype)

norm_probe = RMSNorm(4)
norm_input = torch.tensor([[[1., 2., 3., 4.], [2., 4., 6., 8.]]])
norm_output = norm_probe(norm_input)
print("输入/输出:", norm_input.shape, norm_output.shape)
print("归一化后的均方根:", norm_output.square().mean(-1).sqrt().detach())
torch.testing.assert_close(norm_output[0,0], norm_output[0,1], rtol=1e-5, atol=1e-5)
""")

md(r"""
**观察：** 两个 token 只差一个正比例倍数，归一化后近似相同。输出 shape 不变。

### 5.2 位置从哪里来？RoPE 把 Q、K 旋转

没有位置信息时，注意力很难区分“我爱你”和“你爱我”。RoPE 将每个注意力头的偶数、奇数坐标两两配对，以随位置变化的角度旋转：

$$\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix},\quad
\theta=\mathrm{position}\times10000^{-2i/d_h}.$$

旋转保持向量长度；旋转后的 Q、K 点积会携带相对位置信息。我们对 Q、K 使用 RoPE，V 保留原值。缓存解码时必须从历史 token 数开始计位置，不能每次都从 0 开始。参见 [RoFormer 原论文](https://arxiv.org/abs/2104.09864)。
""")

code(r"""
def apply_rope(x, positions):
    # x: [B,H,T,Dh]；positions: [T]，可以从非零位置开始。
    head_dim = x.shape[-1]
    inv_freq = 10000.0 ** (-torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :].to(x.dtype)
    sin = angles.sin()[None, None, :, :].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1)
    return rotated.flatten(-2)  # 把每对坐标恢复为一个 Dh 维向量

rope_probe = torch.randn(1, 2, 4, 8)
rope_result = apply_rope(rope_probe, torch.arange(4))
torch.testing.assert_close(rope_probe.square().sum(-1), rope_result.square().sum(-1), rtol=1e-5, atol=1e-5)
torch.testing.assert_close(rope_probe[:,:,0], rope_result[:,:,0])
print("位置 0 保持不变；其他位置改变方向但保持长度。")
""")

md(r"""
**练习：** 把所有 `positions` 改成 0，预测结果；再把位置从 10 开始，验证长度仍不变。

### 5.3 Attention：每个位置提出问题，历史位置提供信息

$Q=xW_Q$ 表示“我正在寻找什么”，$K=xW_K$ 表示“我能被怎样匹配”，$V=xW_V$ 表示“我提供什么信息”。这些描述是帮助理解的比喻，三个矩阵都由训练学习。

$$A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\qquad O=AV.$$

**因果掩码 $M$** 在未来位置为 $-\infty$，其他位置为 0。Softmax 后未来权重为 0。若忘记掩码，模型训练时可以偷看答案，loss 会好看，生成却失败。
""")

code(r"""
# 先用一个头、三个位置观察注意力矩阵，不涉及网络类。
q_demo, k_demo, v_demo = [torch.randn(1,1,3,4) for _ in range(3)]
allowed_demo = torch.ones(3,3,dtype=torch.bool).tril()
scores_demo = q_demo @ k_demo.transpose(-2,-1) / math.sqrt(4)
weights_demo = scores_demo.masked_fill(~allowed_demo, float("-inf")).softmax(-1)
output_demo = weights_demo @ v_demo
print("每行对应一个查询位置，每列对应一个被读取的位置:")
print(weights_demo[0,0])
assert torch.equal(weights_demo[0,0].triu(1), torch.zeros(3,3))
torch.testing.assert_close(weights_demo.sum(-1), torch.ones(1,1,3))
print("输出:", output_demo.shape)
""")

md(r"""
### 5.4 多头与实际实现

多头只是把 $d$ 维拆成 $H$ 组，每组做独立注意力，再拼回来。下面的 `manual` 路径正好实现上面的公式；`SDPA` 路径让 PyTorch 选择更省显存的实现，**没有调用预训练 Transformer**。

缓存 `K,V` 的长度为 $P$，新输入长度为 $T$ 时，查询位置为 $P,P+1,\ldots$，键位置为 $0,1,\ldots,P+T-1$。合法条件是 `key_position <= query_position`。缓存场景不可以简单对一个非方阵套左上角三角掩码。

PyTorch SDPA 的布尔 mask 中，`True` 表示允许注意。我们把 `dropout_p` 固定为 0，便于比较路径与检查缓存；`is_causal=True` 只用于没有历史的方阵。接口和 mask 语义见 [PyTorch 2.9 SDPA 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)。
""")

code(r"""
class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.use_sdpa = config.use_sdpa
        self.qkv = nn.Linear(config.dim, 3*config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x, kv_cache=None, use_cache=False):
        batch, length, dim = x.shape
        # 一次投影得到 Q/K/V，然后拆出头维度。
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch,length,self.n_heads,self.head_dim).transpose(1,2) for t in (q,k,v)]
        past_length = 0 if kv_cache is None else kv_cache[0].shape[-2]
        positions = torch.arange(past_length, past_length+length, device=x.device)
        q, k = apply_rope(q,positions), apply_rope(k,positions)
        if kv_cache is not None:
            k = torch.cat((kv_cache[0],k), dim=-2)
            v = torch.cat((kv_cache[1],v), dim=-2)
        key_positions = torch.arange(k.shape[-2], device=x.device)
        allowed = key_positions[None,:] <= positions[:,None]
        if self.use_sdpa:
            if past_length == 0:
                out = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
            else:
                out = F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.0)
        else:
            scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
            # Softmax 在 FP32 计算，之后转回 V 的 dtype。
            weights = scores.float().masked_fill(~allowed,float("-inf")).softmax(-1).to(v.dtype)
            out = weights @ v
        out = out.transpose(1,2).contiguous().view(batch,length,dim)
        new_cache = (k,v) if use_cache else None
        return self.out_proj(out), new_cache

attention_probe = CausalAttention(replace(cfg,use_sdpa=False))
attention_x = torch.randn(1,4,cfg.dim)
attention_y, _ = attention_probe(attention_x)
print("多头前后:", attention_x.shape, "->", attention_y.shape)
""")

md(r"""
**检查点：** Q/K/V shape 是 `[B,H,T,Dh]`；注意力权重是 `[B,H,T,T]`；合并后又是 `[B,T,d]`。不要把 `H` 与 `T` 转置反了。

### 5.5 SwiGLU 与残差：既做信息交换，也做特征变换

Attention 在位置之间交换信息；FFN 则独立处理每个位置的特征。SwiGLU 使用一个分支作为门：

$$\mathrm{FFN}(x)=\big(\mathrm{SiLU}(xW_g)\odot(xW_u)\big)W_d,$$

其中两个升维投影得到 `[B,T,m]`，逐元素相乘，再降回 `[B,T,d]`。残差加法 `x + 分支(x)` 让梯度有更直接的通路。我们用 Pre-Norm：先归一化再进入分支。
""")

code(r"""
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = SwiGLU(config)

    def forward(self, x, kv_cache=None, use_cache=False):
        attention_output, new_cache = self.attn(self.attn_norm(x),kv_cache,use_cache)
        x = x + attention_output  # 第一个残差
        x = x + self.ffn(self.ffn_norm(x))  # 第二个残差
        return x, new_cache

block_probe = TransformerBlock(cfg)
block_y, _ = block_probe(torch.randn(1,4,cfg.dim))
assert block_y.shape == (1,4,cfg.dim)
print("一个 Transformer Block 保持形状:", block_y.shape)
""")

md(r"""
### 5.6 把部件组装成语言模型

最后一层输出 $h$ 后，乘 Embedding 参数的转置得到词表 logits：$z=hE^\top$。同一个词向量既负责读 token，也负责与输出特征匹配，这就是权重共享。

完整配置会开启**激活重计算**：反向传播时重新执行 block 前向，减少训练中保存的激活，代价是额外计算。只有训练且需要梯度时启用；推理缓存不走这条路径。它与“保存模型的 checkpoint 文件”是两件事。
""")

code(r"""
class TinyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(config.vocab_size,config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim,config.vocab_size,bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embedding.weight  # 同一个 Parameter，仅统计与优化一次

    def _init_weights(self, module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self, ids, use_cache=False, kv_cache=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("输入应为非空 [batch,time] 整数 token")
        if kv_cache is not None and len(kv_cache) != len(self.blocks):
            raise ValueError("缓存层数与模型层数不符")
        past = 0 if kv_cache is None else kv_cache[0][0].shape[-2]
        if past + ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("超出上下文长度；聊天时请先裁剪完整历史轮次")
        x, new_cache = self.embedding(ids), []
        for index, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[index]
            if self.cfg.gradient_checkpointing and self.training and torch.is_grad_enabled() and not use_cache and kv_cache is None:
                # 直接传当前 block 的绑定方法，避免循环闭包在反向时引用最后一层。
                x, _ = checkpoint(block,x,use_reentrant=False)
                cache = None
            else:
                x, cache = block(x,layer_cache,use_cache)
            if use_cache:
                new_cache.append(cache)
        logits = self.lm_head(self.final_norm(x))
        return logits, (new_cache if use_cache else None)

model = TinyLLM(cfg).to(DEVICE)
actual_parameters = sum(p.numel() for p in model.parameters())
assert actual_parameters == parameter_budget(cfg)
assert model.lm_head.weight is model.embedding.weight
print("实际可训练参数:", f"{actual_parameters:,}", "公式一致；输出权重共享。")
""")

md(r"""
### 5.7 最关键的模型自检：看不到未来，梯度能传回去

先拿两条前缀相同、后缀不同的输入。由于因果掩码，公共前缀位置的 logits 应相同。然后做一次反向传播，检查 loss 和梯度都是有限数。

这一步比“模型运行没有报错”更有意义：一个忘了因果掩码的模型同样能运行。
""")

code(r"""
model.eval()
ids_a = torch.tensor([[BOS_ID]+tokenizer.encode("学习语言模型")[:12]],device=DEVICE)
ids_b = ids_a.clone()
prefix_length = max(1,ids_a.shape[1]//2)
ids_b[:,prefix_length:] = (ids_b[:,prefix_length:]+1) % cfg.vocab_size
with torch.no_grad():
    logits_a, _ = model(ids_a)
    logits_b, _ = model(ids_b)
torch.testing.assert_close(logits_a[:,:prefix_length],logits_b[:,:prefix_length],rtol=2e-4,atol=2e-5)

# 所有位置都预测下一个 token：此时 x 和 y 已经错开，不再重复 shift。
model.train()
ids_probe = torch.tensor(np.array(train_tokens[:17],dtype=np.int64),device=DEVICE).unsqueeze(0)
logits_probe, _ = model(ids_probe[:,:-1])
loss_probe = F.cross_entropy(logits_probe.reshape(-1,cfg.vocab_size),ids_probe[:,1:].reshape(-1))
loss_probe.backward()
assert torch.isfinite(loss_probe)
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
print("因果性通过；loss:", round(loss_probe.item(),4), "Embedding梯度范数:", model.embedding.weight.grad.norm().item())
model.zero_grad(set_to_none=True)  # 自检不是一次训练，清理其梯度
""")

md(r"""
**练习：** 暂时把手写 Attention 的 mask 去掉，再跑公共前缀测试，观察失败。恢复 mask 后再继续。你的测试现在能捕捉一种真实的模型错误。

## 6. 自回归生成与 KV Cache

训练可并行预测很多位置；生成必须一步一步：读 prompt → 看最后一个位置的 logits → 选择一个 token → 接回输入 → 再预测。

缓存保留每层历史 token 的 K、V。新 token 只算自己的 Q/K/V，然后读取旧 K/V；历史 token 的隐藏状态不需要每次重算。这不改变模型的预测含义，但减少重复工作。
""")

code(r"""
model.eval()
cache_ids = ids_probe[:,:12]
with torch.no_grad():
    full_logits, _ = model(cache_ids)
    first_logits, cache = model(cache_ids[:,:8],use_cache=True)
    # 一次追加4个token，检查非方阵因果mask与RoPE位置偏移。
    next_logits, cache = model(cache_ids[:,8:],use_cache=True,kv_cache=cache)
    cached_logits = torch.cat((first_logits,next_logits),dim=1)
torch.testing.assert_close(full_logits,cached_logits,rtol=2e-4,atol=2e-5)
print("缓存与全量 logits 一致；每层缓存 K shape:", cache[0][0].shape)
""")

md(r"""
### 6.1 怎样选下一个 token？

- 贪心：取最高 logit，方便可重复的评估。
- Temperature：logits 除以正数温度；小于 1 更集中，大于 1 更分散。
- Top-k：只保留概率最高的 k 个候选。
- Top-p：保留累计概率刚达到 p 的最小候选前缀。

Top-k/top-p 改变采样分布。聊天可以用它们，后面 RL 为了正确计算策略概率，将固定温度 1 且不截断。EOS 停止生成；`max_new_tokens` 防止永不结束。
""")

code(r"""
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=1.0):
    # logits 为 [1,V]；返回 [1,1]，与输入的 batch 维保持一致。
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature >= 0，且 0 < top_p <= 1")
    if temperature == 0:
        return logits.argmax(dim=-1,keepdim=True)
    scores = logits.float() / temperature
    if top_k is not None:
        if not isinstance(top_k,int) or top_k < 1:
            raise ValueError("top_k 必须为正整数或 None")
        k = min(top_k,scores.shape[-1])
        cutoff = scores.topk(k,dim=-1).values[:,-1,None]
        scores = scores.masked_fill(scores < cutoff,float("-inf"))
    if top_p < 1:
        sorted_scores, order = scores.sort(dim=-1,descending=True)
        cumulative = sorted_scores.softmax(-1).cumsum(-1)
        remove = cumulative > top_p
        remove[:,1:] = remove[:,:-1].clone()  # 保留首次超过阈值的那个 token
        remove[:,0] = False
        sorted_scores = sorted_scores.masked_fill(remove,float("-inf"))
        scores = torch.full_like(scores,float("-inf")).scatter(-1,order,sorted_scores)
    return torch.multinomial(scores.softmax(-1),num_samples=1)

@torch.no_grad()
def generate(policy, prompt_ids, max_new_tokens=32, temperature=1.0, top_k=None, top_p=1.0, use_cache=True):
    if not prompt_ids or max_new_tokens < 0:
        raise ValueError("prompt 不为空；max_new_tokens >= 0")
    if len(prompt_ids)+max_new_tokens > policy.cfg.max_seq_len:
        raise ValueError("prompt + 生成预算超出上下文；请缩短输入或生成预算")
    was_training = policy.training
    policy.eval()
    device = next(policy.parameters()).device
    ids = torch.tensor([prompt_ids],dtype=torch.long,device=device)
    result, cache = list(prompt_ids), None
    try:
        for _ in range(max_new_tokens):
            # 第一步读整个 prompt，之后缓存路径只读刚生成的 token。
            with amp_context():
                logits, cache = policy(ids,use_cache=use_cache,kv_cache=cache)
            token = sample_next_token(logits[:,-1],temperature,top_k,top_p)
            result.append(int(token.item()))
            if result[-1] == EOS_ID:
                break
            ids = token if use_cache else torch.tensor([result],dtype=torch.long,device=device)
    finally:
        policy.train(was_training)
    return result

untrained_prompt = [BOS_ID]+tokenizer.encode("语言模型")
untrained_output = generate(model,untrained_prompt,max_new_tokens=8,temperature=0)
print("尚未训练时的续写:", repr(tokenizer.decode(untrained_output)))
""")

md(r"""
**观察：** 随机参数的输出可能是乱码、重复符号或不通顺的词片段，这很正常。架构本身不会自带中文知识。

**练习：** 设 `top_k=1` 与贪心相比，结果应该怎样？如果模型立刻输出 EOS，空回答能否直接证明生成代码有错？需要结合训练与 logits 分析。

到这里你已经实现了一个可以前向、反向、带缓存生成的完整语言模型。接下来，学习数据如何把这个概率模型训练成基座模型，再训练成指令模型。
""")
