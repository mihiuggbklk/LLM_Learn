# 真实语料与数据格式

首次学习无需任何外部数据，Notebook 的 `smoke` 路线自带离线教学语料。

## 下载英文维基百科，开始 full

若 `full` 报 `FileNotFoundError: 请先在 ... data/raw 放 txt/jsonl 真实语料`，表示本地尚未准备预训练数据。在项目根目录执行：

```powershell
python tools/download_corpus.py --max-rows 5000
```

脚本只用 Python 标准库，通过 Hugging Face 官方数据接口按页下载，不需要安装 `datasets`。下载完成后生成 `data/raw/wikipedia_20231101_en.jsonl`，Notebook 会自动读取。保留 `PROFILE="full"`，重启内核后从头运行；确认实际词表为 **8192**、参数量为 **98,324,736**。

来源是 [Wikimedia 官方 Wikipedia 数据集](https://huggingface.co/datasets/wikimedia/wikipedia)，选择 [20231101.en 英文快照](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en)。它由维基百科文章清洗而来，不是本项目生成的模板或合成问答；不能据此保证每篇文章都没有机器编辑。每行保存完整 `text`、原始 `id`、`title`、`url` 和 `source`，训练读取器只使用 `text`。

- 英文全集有 **6,407,814** 篇文章，41 个 Parquet 文件合计约 **11.6 GB**。本脚本默认仅取前 5,000 行，不下载整个库。
- 这是按数据集顺序取出的起步子集，不是随机抽样或平衡语料。保留英文原文、不拆分文章，以便 Notebook 按完整文档清洗、精确去重和划分。
- 来源、快照、下载范围、实际行数及 SHA256 保存在 `data/sources/wikipedia_20231101_en.json`。接口响应缓存放在 `data/download_cache/`，中断后可重跑，已缓存的页不用重复下载。最终文件就绪前不发布 `.jsonl`，避免训练误读半份数据。
- 数据卡标注 **CC BY-SA 3.0 / GFDL**，并说明部分文本仅适用 Creative Commons；归属及具体条款见[数据卡](https://huggingface.co/datasets/wikimedia/wikipedia#licensing-information)和 [Wikimedia 许可说明](https://dumps.wikimedia.org/legal.html)。保留文章来源链接和来源清单。

扩大到前 20,000 行时执行：

```powershell
python tools/download_corpus.py --max-rows 20000 --force
```

`--force` 会替换脚本生成的同名语料和来源清单。更换语料后需重新运行数据划分和分词器训练；新分词器不能与旧模型 checkpoint 混用。几千篇真实文章足以用于起步验证，不代表已准备好充分训练约一亿参数模型的数据预算。

本次按要求使用英文：先前下载的中文文件已移到 `data/archive/`，不参与 `data/raw/` 的训练输入。Notebook 的中文教学文字和代码说明不影响预训练语料的语言。

## 放入其他预训练数据

切换到 `full` 前，建立 `data/raw/`，放 UTF-8 的 `.txt` 或 `.jsonl`：

- txt 每个非空行是一篇文档。
- jsonl 每行 `{"text":"一篇完整文档……"}`。
- 预训练前按文档清洗、去重、划分；正式评估还需要同源与近重复去污染。
- 分词器只在训练文本上学习，完整目标词表需要达到 8192。

## SFT 对话数据

维基文章是预训练文本，不会自动成为指令对话。当前下载脚本只准备预训练数据；未提供 SFT 文件时，Notebook 后续仍使用原来的算术教学样本。

自己的 SFT 数据分别放 `data/sft_train.jsonl` 与 `data/sft_val.jsonl`，每行：

```json
{"messages":[{"role":"user","content":"What is gradient descent?"},{"role":"assistant","content":"Gradient descent updates parameters in the direction that reduces the loss."}]}
```

可以在开头加入 system 消息，也可以有多轮 user/assistant。训练/验证问题应独立；本教程的合成题只用于演示训练机制。

上面的英文问答只展示文件格式，不属于下载的真实语料。接入真实英文 SFT 数据时，需检查当前 256-token 上下文限制、来源许可和训练/验证重叠。
