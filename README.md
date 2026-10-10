# 从零设计并实现约 0.1B LLM

主要教材是 [0.1B_LLM.ipynb](./0.1B_LLM.ipynb)。内容以教学为中心：中文原理、张量形状、公式、逐步实现、中文代码注释、可观察自检与练习。

从字节级 BPE、数据处理、RMSNorm/RoPE/因果多头注意力/SwiGLU 开始，手写共享权重的 Transformer、预训练、SFT、RLVR、KV Cache 和多轮对话。不加载预训练模型参数。

完整模型：8192 词表、13 层、768 隐藏维度、12 头、2048 FFN，**98,324,736 参数**。默认 `smoke` 用相同架构的小配置和离线小语料进行短实验；`full` 使用自己的真实语料。短实验不代表完成完整训练，也不会获得 ChatGPT 级能力。

## 开始学习

本地已检测到 RTX 4060 Laptop GPU，8 GiB 显存。教程验证环境使用 PyTorch 2.9.1+cu130；GPU PyTorch 的安装请按 [官方选择器](https://pytorch.org/get-started/locally/) 匹配平台和驱动。已有可用的 PyTorch 环境无需重装。

```powershell
python -m pip install -r requirements.txt
python -m jupyter lab
```

在 Jupyter 中打开教材、选择本机已有的 **Python (pytorch)** 内核，从上到下执行。先保持 `PROFILE="smoke"`。模型部分可先单独学习；理解预训练和 SFT 后再进入 RL。

换到其他电脑时，选择该机安装了 PyTorch 的内核即可。需要注册当前环境时执行 `python -m ipykernel install --user --name pytorch --display-name "Python (pytorch)"`。验证脚本也支持环境变量 `LLM_NOTEBOOK_KERNEL` 指定该机的内核名称。

## 完整 0.1B 路线

没有本地语料时，先在项目根目录执行 `python tools/download_corpus.py --max-rows 5000`，下载官方英文维基百科真实文章到 `data/raw/`。具体来源、许可、扩大样本与 SFT 的区别见[数据说明](./data/README.md)。

按 [数据说明](./data/README.md) 放入真实语料，重启内核，把控制面板 `PROFILE` 改为 `full`。先确认词表与参数，再做少量训练更新；根据实测显存、吞吐和验证损失增加预算。8 GiB 从 context 256、micro batch 1、AMP、SDPA、激活重计算开始。SFT 与 RL 依次建立在有任务能力的模型之上。

Notebook 第 11 章给出 token 预算、恢复训练与阶段学习项目。`artifacts/smoke/`、`artifacts/full/` 保存各路线分词器、token 文件和训练产物。

## 验证与维护

```powershell
# 重新生成教材（会清除旧 cell 输出）
python tools/build_notebook.py
# 顺序执行默认短实验、关键行为检查、完整模型的一步 CUDA 更新
python tools/validate_notebook.py --execute --full-probe
# 单独运行关键行为检查
python tests/test_notebook.py
```

[validation_report.json](./validation_report.json) 记录实际验证范围。完整模型探针只训练一次随机 token 更新，验证实现和资源；没有完成长期语料训练或证明对话质量。
