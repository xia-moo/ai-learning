# 中文情感分类与推理评估练习

基于 BERT 的负面、中性、正面三分类项目，包含继续微调、数据补充对照实验、FastAPI 服务、批量推理、INT8 动态量化和处理规则评估。

本文依据 2026-11-06 的本地源码、环境及已保存报告整理。项目保留了学习过程中的脚本，不是生产部署模板。

## 快速启动

在 Windows PowerShell 中使用已有环境：

```powershell
conda activate AIDocsAgent_py312
Set-Location D:\workprojects\ai-learning
python -m uvicorn sentiment_api:app --app-dir "D:\workprojects\ai-learning" --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000/docs 。根路径 `/` 没有注册，返回 404 属于预期。

当前源码配置：

| 项目 | 值 / 设置位置 |
|---|---|
| 检查点 | `predict_sentiment.py` 的 `checkpoint_path`：`bert_sentiment_negation_v1.pt` |
| 推理模式 | `USE_INT8 = False`，FP32，CPU |
| 推理长度上限 | `INFERENCE_MAX_LENGTH = 64`，包括特殊 token |
| 处理阈值 | `sentiment_api.py` 的 `DECISION_THRESHOLD = 0.90` |
| 类别顺序 | 负面、中性、正面 |
| 句子表示 | 排除 PAD 的 masked mean pooling |

修改模型路径、模式、长度或阈值后，需要停止并重启服务。训练使用原检查点的长度上限 32；服务评估固定使用 64。前后比较时分别保持这些条件一致。

运行依赖本地检查点及已缓存的 `google-bert/bert-base-chinese` tokenizer/config。当前加载采用 `local_files_only=True`；复制 Python 文件到另一台电脑并不包含这些资源，也不自动下载它们。

GitHub 仓库不包含模型权重（`*.pt` 等）、下载缓存和运行时批量统计。首次使用需要准备基础模型资源，并运行 `learn_transformer.py` 生成原检查点，再按需要运行 `train_negation.py` 或 `train_control.py` 生成对应的微调检查点。训练会消耗计算资源；已有本地检查点时无需为上传代码而重新训练。

## 接口与处理规则

单条：`POST /predict`

```json
{"text":"好用"}
```

批量：`POST /predict/batch`

```json
{"texts":["好用","我买的是白色版本"]}
```

每条结果返回 `text`、`label`、`probabilities`、`max_length`、`original_token_count`、`input_token_count`、`truncated`、`inference_mode`、`checkpoint_sha256`、`decision_threshold`、`action`、`reason`。批量响应外层包含 `count` 与 `results`，保持清洗后的输入顺序。

规则由 `sentiment_policy.py` 统一执行：发生截断先转人工；未截断且正面概率达到阈值才自动通过；其他情况转人工。处理动作与模型预测类别分别记录。“转人工”只是返回标记，尚未连接真实审核队列。阈值 0.90 是教学实验配置，不代表已满足真实业务质量要求。

空白文本返回 400；缺少必填字段返回 422。有效 token 数不包含 PAD。

## 常用入口

在项目目录下执行。标记“服务”的脚本需要另一个终端保持 API 运行。

| 文件 | 用途 |
|---|---|
| `sentiment_api.py` | 由 Uvicorn 启动服务 |
| `predict_sentiment.py` | 加载模型、单条及批量推理、分组和性能练习 |
| `sentiment_policy.py` | 概率阈值和截断处理规则 |
| `call_sentiment_api.py` | 单条 HTTP 请求（服务） |
| `test_sentiment_policy.py` | 不加载模型的规则检查 |
| `test_sentiment_api.py` | 200/400/422 接口检查（服务） |
| `test_truncation_api.py` | 单条与批量截断处理检查（服务） |
| `evaluate_sentiment_api.py` | 开发集指标、阈值及假设代价比较（服务） |
| `evaluate_holdout.py` | 固定配置的新增人工样例检查（服务） |
| `compare_models.py` | 读取旧/新报告，比较分类及处理动作变化 |
| `compare_checkpoint_weights.py` | 比较两个文件的参数；先检查脚本中的目标路径 |
| `summarize_batches.py` | 请求两批样例，覆盖最新统计并追加历史（服务） |
| `read_summary.py`、`read_history.py` | 读取最新统计、历史及配置筛选 |
| `train_negation.py` | 从原检查点开始，38 条数据训练 30 次更新 |
| `train_control.py` | 从原检查点开始，仅原 30 条数据训练 30 次更新 |
| `benchmark_sentiment_api.py`、`benchmark_batch_api.py` | HTTP 延迟、吞吐量和并发实验（服务） |
| `quantize_sentiment.py` | FP32 / INT8 本地比较；检查其独立模型路径和长度设置 |

其余 `learn_*.py`、`pytotch.py`、pandas 等文件保留为课程练习。`learn_transformer.py` 含原始训练流程，运行会保存原检查点，不应作为启动服务的入口。

## 检查与报告

不请求 API 的检查：

```powershell
python test_sentiment_policy.py
python compare_models.py
```

API 运行后：

```powershell
python test_sentiment_api.py
python test_truncation_api.py
python evaluate_sentiment_api.py
```

`evaluate_holdout.py` 的首次报告已经存在。它会拒绝覆盖 `sentiment_holdout_report.json`，这是预期保护。不要为了重复执行而删除原报告；后续实验应明确新数据/配置并使用新的报告路径。

开发评估脚本保留了教学阶段的固定报告文件名，包括 `sentiment_baseline_before.json` 和 `sentiment_before_dev10.json`。已有文件会保留；如需保存新实验，先修改报告路径与描述，避免把新模型写成旧模型报告。`sentiment_dev.jsonl` 为反复检查和调参使用的开发数据。

## 实验结果及证据

### 否定表达补充训练

原训练数据 30 条，每类 10 条；加入 8 条人工否定表达对照后共 38 条，负面 14、中性 10、正面 14。两组均从 `bert_sentiment_best.pt` 开始，使用 AdamW，BERT 学习率 1e-5、分类层 1e-4、weight decay 0.01，batch size 4，训练随机种子 42。比较起点和第 10、20、30 次更新，在反复使用的 6 条验证样例上按损失选择。

| 实验 | 选择结果 | 验证损失 |
|---|---|---:|
| 原模型 | 起点 | 0.2592 |
| 仅原数据继续训练 | 第 0 步，恢复原参数 | 0.2592 |
| 加入否定样例 | 第 30 步（第 3 轮） | 0.1930 |

对照组与原模型参数逐项比较全部一致；增强模型的 199 个 BERT 状态张量中 197 个发生变化，分类层 2 个张量均变化。参数变化数不等同于性能提升幅度。

两组更新次数一致，样本访问次数、批次组成及计算量并不完全一致。只做了一次固定种子的微型实验，不能由此推断稳定的因果效果。

同一 10 条开发样例、FP32、长度 64、阈值 0.90 的保存报告：

| 指标 | 原模型 | 增强模型 |
|---|---:|---:|
| 分类正确数 | 9/10 | 10/10 |
| 自动通过数 | 1/10 | 4/10 |
| 误放行数 | 0 | 0 |
| 正面自动通过召回率 | 25% | 100% |

证据：`sentiment_before_dev10.json`、`sentiment_after_dev10.json`。修正了“这个产品并没有让我失望。”的一次分类错误，原来正确的 9 条仍正确。另两条新增自动通过源于概率跨越阈值，并非修正分类错误。

### 新增人工样例检查

固定增强模型、FP32、长度 64、阈值 0.90，在事先编写并标注的 12 条样例上检查：分类正确 12/12，自动通过 4/12，无误放行、无截断。通过样本精准率 4/4，正面自动通过召回率 4/4。

证据：`sentiment_holdout.jsonl`、`sentiment_holdout_report.json`，报告包含数据及检查点 SHA-256。收尾时已核对报告计数与磁盘数据/模型哈希一致，未重复请求或覆盖报告。

这些是人工编写的小型诊断样例，不能代表真实流量上的整体准确率，也不足以证明业务精准率达到 95%。已查看并用于后续决策的样例不再提供全新的盲测证据。

## 量化与性能实验的范围

曾完成线性层动态 INT8 量化、按长度分批、分词复用、阶段计时和 HTTP 并发测试。量化模式下观察到同一文本随批次组成出现概率差异；因此当前最终检查使用 FP32。旧检查点的性能测量不能直接视为增强模型的实测结果。报告性能时应同时记录检查点、推理模式、输入长度/批次、硬件、并发和计时范围。

## 数据与文件约定

- `sentiment_negation_train.jsonl`：8 条训练补充样例。
- `sentiment_dev.jsonl`：10 条开发样例，不是独立测试集。
- `sentiment_holdout.jsonl`：12 条新增人工检查样例。
- `batch_summary.json`：最近一次统计，覆盖写入。
- `batch_summary_history.jsonl`：每次脚本运行追加一条记录；累计量是处理条次，不是不同评论数。
- 旧记录的 `max_lengths` 与新字段 `inference_max_lengths` 含义相同；缺失模式/模型哈希的旧记录不能猜测补齐。
- 检查点 SHA-256 标识文件字节，重新序列化或增加元数据即可改变哈希；比较参数是否相同需逐张量检查。
- `bert_sentiment_best.pt` 为原模型；`bert_sentiment_negation_v1.pt` 为增强模型；`bert_sentiment_control_v1.pt` 为对照选择结果。后两者用创建模式保护，不覆盖已有文件。

## 环境与当前限制

本机环境 `AIDocsAgent_py312`，Python 3.12.15。`requirements-sentiment.txt` 记录本机主要直接依赖的已安装版本；未在全新环境重新安装验证，也不是包含传递依赖、操作系统和模型缓存的完整锁文件。已有环境运行无需重新安装。新环境需另行准备合适的 PyTorch 构建、检查点与 tokenizer/config 缓存。

目前 `predict_sentiment.py` 仍有顶层演示推理，导入服务时会额外执行并打印；分组/性能练习与服务函数仍处于同一文件。部分训练、读取、评估脚本在模块顶层执行，不能随意导入复用。数据量小、尚无代表性真实数据评估、并发负载限制和完整部署运维流程。因此本项目适合作为可演示的学习与实验项目，后续工程整理应先隔离演示入口，再验证行为一致性。
