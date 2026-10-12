# RCUF：混合模型指令寿命与状态恢复的生死实验

本仓库实现一个无需训练的分阶段先导实验：**自然语言策略是否漂移 → 循环状态恢复是否保住内容 → 一次恢复能维持多远**。默认模型是 `Qwen/Qwen3.5-4B`。`Qwen3-4B` 是纯 Transformer，不能运行这里的 GDN 状态干预。

两张 A100 80GB 各加载一份模型，按**文档**分工；同一文档的全部方法在同一 GPU 上从独立缓存副本开始。没有 tensor parallel 或 DDP。单卡可使用 `--devices 0`。

运行环境参考 [GiangW1/Grace](https://github.com/GiangW1/Grace) 的 Python 3.11、BF16、绑卡、tmux 和分阶段检查方式。Grace 公开配置没有固定完整服务器依赖锁，因此这里建立独立环境，并记录实际包版本、模型 revision、数据 revision、源码和运行指纹。

**实现状态：代码包含 CPU 数学和实际小型 HF 模型缓存测试。4B 模型的 A100/FLA 实测需要在服务器上执行。测试通过不代表自然漂移或恢复收益已经成立。**

## 1. 新建服务器环境

在 tmux 中执行；保留已有 Grace 环境。

```bash
conda create -n rcuf python=3.11 -y
conda activate rcuf
git clone https://github.com/GiangW1/rcuf.git
cd rcuf

# CUDA 12.6 wheel；先用 nvidia-smi 确认驱动支持该运行时。
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[gpu,dev]'
python -m pytest
python scripts/doctor.py
```

默认固定 `transformers==5.19.0`、`flash-linear-attention==0.5.2`、`torch==2.8.0`。FLA 通过官方 gated-delta-rule 函数接入 HF 的 GDN 计算；attention 使用 PyTorch SDPA，卷积保留 HF 实现。无需 vLLM、verl、flash-attn 或 causal-conv1d 的 CUDA 编译。

数据依赖另固定 `datasets==4.0.0`、`pyarrow==20.0.0`、`pandas==2.2.3` 和 `numpy==1.26.4`，避免新版 Arrow 的 NumPy 2 要求与语言识别器混装。

如果服务器已经有可用 PyTorch，请先在新环境确认版本。不要直接把本项目安装进 Grace 后训练环境。

本机 CPU 开发检查：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[cpu,dev]'
pytest
```

## 2. 准备并冻结数据

```bash
rcuf prepare --config configs/default.json --data-dir data/pilot
```

默认流式读取 `wikimedia/wikipedia` 的 `20231101.en` 英文资料，构造 8 份开发文档和 32 份独立测试文档。不同文档使用不重叠文章；同一文档的 0/8K/32K 条件保持相同随机记录和问题。源文章标题、URL 和文本哈希保存在数据中。维基文本受其源许可约束，生成的数据和模型权重均不上传本仓库。

`prepare` 将 `main` 解析为实际 commit SHA，并写入 `manifest.json`。之后所有阶段读取冻结的配置和 token 流，修改配置文件不会暗中改变已经准备的数据。重新设定实验请使用新数据目录和新运行目录。完整输入由官方非思考模板一次性渲染和 tokenize；切分点通过字符 offset 映射到 token，不拼接分别 tokenize 的提示片段。

也可使用你已有的普通英文资料，每行一个 JSON 对象：

```json
{"text":"A sufficiently long ordinary English document...", "title":"source title", "url":"source URL"}
```

```bash
rcuf prepare --config configs/default.json --data-dir data/local-pilot --corpus-jsonl /path/to/corpus.jsonl
```

每行文本至少 500 字符。资料不足、重复、包含特殊角色 token 或明显改写系统指令的语句会被拒绝/跳过。过滤不是完善的注入检测器；建议人工检查资料来源。第一轮英文资料属于语言不同的背景数据，不应称为恶意冲突指令。

## 3. 两卡短检查

可以先准备很小的资料集，仅检查运行链：

```bash
rcuf prepare --config configs/smoke.json --data-dir data/smoke
rcuf launch --data-dir data/smoke --run-dir runs/smoke --stage smoke --devices 0,1
```

正式实验也会重复缓存检查：

```bash
tmux new -s rcuf
rcuf launch --data-dir data/pilot --run-dir runs/pilot --stage smoke --devices 0,1
```

每张卡运行一个短文档的基线、低秩相加、投影恢复。每个 worker 还检查：

- 同一缓存独立分叉的 logits 完全复现；
- 多 token 缓存续接与整体预填接近，首 token 一致；
- 修改循环状态确实影响重新计算的问题 logits；
- 长状态为 FP32，布局为 `[batch, value_head, key_dim, value_dim]`；
- 提取法语/英语参考状态和每头 rank-4 能量。

日志位于 `runs/pilot/smoke/worker-0.log`、`worker-1.log`。失败时先读日志。启动器会停止另一个 worker，避免继续花算力。

**较早的 Transformers Qwen3.5 实现会在多 token 缓存续接中忽略已有 GDN 状态。禁止把这里的版本要求删掉，或直接降到 5.3.0。**

## 4. 第一关：自然漂移，96 次回答

```bash
rcuf launch --data-dir data/pilot --run-dir runs/pilot --stage baseline --devices 0,1
rcuf summarize --data-dir data/pilot --run-dir runs/pilot --stage baseline
```

系统策略是“全部回答使用法语”；回答任务是末尾记录中的授权码和预算，要求两句完整回答。默认测试 32 文档 × 3 长度。

指标：`language_ok`（法语遵循）、`content_ok`（两个字段都正确且无矛盾）、`joint_ok`（二者同时正确）。编号、数字和专名从语言识别输入中移除。内容匹配规范化千位分隔符；严格解析无法判断的语义需要人工复核。

默认投入门槛：短条件语言和内容各 ≥31/32；32K 语言净损失 ≥7/32；32K 内容 ≥29/32。这些是人为投资门槛。结果没有达到门槛时，当前配置停止，不能外推为所有混合模型的否定结论。

### 人工盲审

自动语言评分可能误判。短/混合/低置信/截断回答，以及同一文档在方法或长度间发生语言判定变化的结果，必须复核。导出文件隐藏方法、长度和参数：

```bash
rcuf review --data-dir data/pilot --run-dir runs/pilot --stage baseline --output runs/pilot/baseline-review.jsonl
```

将每行的 `language_ok`、`content_ok` 从 `null` 填为 JSON `true`/`false`。法语必须有可判断的自然语言回答；单列代码和数字不算。保留 `cell_id` 和 `output_sha256`。`notes` 可以填写判定理由。也可添加 `--all` 复核整个阶段。

```bash
rcuf summarize --data-dir data/pilot --run-dir runs/pilot --stage baseline --reviews runs/pilot/baseline-review.jsonl
```

结果包括 `summary.json` 和 `scored-results.jsonl`。只有完整且经过必要复核的 `continue_pilot` 才允许启动第二关；未完成结果不会被算成失败样本。

## 5. 开发集定参数，再冻结

第一关阳性后执行：

```bash
rcuf launch --data-dir data/pilot --run-dir runs/pilot --stage tune --devices 0,1
rcuf review --data-dir data/pilot --run-dir runs/pilot --stage tune --output runs/pilot/tune-review.jsonl
# 完成盲审文件后
rcuf select --data-dir data/pilot --run-dir runs/pilot --reviews runs/pilot/tune-review.jsonl
```

开发阶段 160 次短回答：8 文档 × 2 长度 ×（基线 + 三个方法各三个强度）。C/D 相加强度为 0.25/1/4；E 恢复强度为 0.25/0.5/1。rank=4 固定，全部 GDN 层/头，不进行头搜索。优先选择开发集未观察到短任务损伤的配置，再比较长条件联合成功和内容成功，平局选择较小强度。

`selection.json` 仅由开发集生成，绑定数据和源码哈希。已有不同的冻结选择不能被静默覆盖。如果所有候选都损伤短任务，审计会明确标记，测试阶段仍可检验已选配置的局限。

## 6. 第二关：六方法，新增 320 次回答

```bash
rcuf launch --data-dir data/pilot --run-dir runs/pilot --stage compare --devices 0,1
rcuf review --data-dir data/pilot --run-dir runs/pilot --stage compare --output runs/pilot/compare-review.jsonl
# 完成盲审文件后
rcuf summarize --data-dir data/pilot --run-dir runs/pilot --stage compare --reviews runs/pilot/compare-review.jsonl
```

| 方法 | HF 状态布局下的操作 |
|---|---|
| baseline | 原样继续 |
| reminder | 问题前插入短法语提醒 |
| add_full | `S + gamma * (S_F - S_E)` |
| add_lowrank | `S + gamma * U_r U_r^T (S_F - S_E)` |
| replace | `S + lambda * U_r U_r^T (S_F - S)` |
| wrong_anchor | `S + lambda * U_r U_r^T (S_E - S)` |

HF 的 key 维在矩阵左侧，所以取差分状态的**左奇异向量**。这与原 idea 使用 `[value_dim,key_dim]` 时的右侧投影等价。只有 `lambda=1` 严格幂等；较小 lambda 为软恢复。子空间只是候选行为子空间，没有被证明是策略专属。

每头最多 rank=4；数值上为零的奇异方向会被屏蔽并记录实际秩，避免零差分的任意 SVD 方向破坏无关状态。内容自动匹配失败的回答也进入人工复核，允许纠正法语数字拼写等解析遗漏。

干预发生于记录结束、最终问题开始之前。状态分支只直接改变 GDN 长状态；问题、角色标记和非思考 assistant 前缀在干预后重新前向，首 token 不复用旧 logits。文本提醒是一条额外 token 续接流，其 token、时间和缓存增长单独计入。提醒分支可以复用资料前缀。

比较阶段有 384 条结果，其中 64 个端点基线来自第一关的复核后结果，真正新增 320 次回答。正式即时测试累计 416 次。

默认恢复门槛：32K 联合成功净增 ≥7/32；32K 内容和短条件联合成功各净损失 ≤1/32。同时报告投影恢复对低秩相加、文本提醒、错误语言锚点的配对差异和 bootstrap 区间。恢复门槛通过本身不能证明替换优于相加，少量净增应判为不确定。

## 7. 第三关：512/4096 token 延迟，新增 128 次回答

第二关阳性后执行：

```bash
rcuf launch --data-dir data/pilot --run-dir runs/pilot --stage delay --devices 0,1
rcuf review --data-dir data/pilot --run-dir runs/pilot --stage delay --output runs/pilot/delay-review.jsonl
# 完成盲审文件后
rcuf summarize --data-dir data/pilot --run-dir runs/pilot --stage delay \
  --reviews runs/pilot/delay-review.jsonl --compare-reviews runs/pilot/compare-review.jsonl
```

只比较低秩相加和投影恢复，干预点在问题前 512 或 4096 token。完整输入不变；干预后的文本重新前向。两个延迟的原始前缀按位置渐进预填，各方法从该位置的独立快照开始。参数仍使用冻结的开发集选择。

判定一次恢复的持续距离：4096-token 后保留至少一半即时联合收益，并继续满足内容损伤限制。即时收益为零/负时不能宣称“保留一半”。报告延迟条件下对低秩相加的差异。它不测试周期维护，也不直接判定注意力回读或循环自维持机制。

## 8. 断点续跑与结果文件

原命令重新执行即可恢复，保持同一 `--devices` 数量和文档分片。完成的 cell 跳过；只有最后一行未写完的 JSONL 会恢复。完整但损坏的记录、重复 cell、数据变化、源码变化、参数变化或分片变化都会报错。改变 GPU 数量请新建运行目录。

| 文件 | 内容 |
|---|---|
| `data/pilot/manifest.json` | 实际模型/数据 revision、配置、tokenizer 和数据哈希 |
| `stage/results-*.jsonl` | 每次回答、token、自动评分、干预范数、配对标识 |
| `stage/cache-checks-*.json` | 实际缓存续接和状态干预检查 |
| `stage/anchor-audit-*.json` | 每层每头子空间能量和差分范数 |
| `stage/environment-*.json` | 包版本、CUDA、GPU、源码/模型配置指纹 |
| `stage/costs-*.jsonl` | 每次启动的实测总时间、实际资料预填 token/时间 |
| `stage/summary.json` | 完整性、待审样本、配对转换、CI、投入门槛 |
| `selection.json` | 开发集冻结参数和选择审计 |

`prefix_seconds_shared` / `snapshot_seconds` 会在同一快照的各方法结果中重复出现，**不能逐行相加作为总成本**。真实预填成本读取 `costs-*.jsonl`。worker 总时间包含加载、参考状态、检查和分支成本；两卡墙钟与两卡 GPU 秒需要区分。中断启动的成本日志可能缺失，应结合 worker 日志报告。

## 9. 数量和结论边界

| 阶段 | 默认回答数 |
|---|---:|
| smoke | 6（不计正式测试） |
| baseline | 96 |
| tune | 160（开发集） |
| compare | 新增 320，端点基线复用 |
| delay | 128 |
| 正式独立回答总数 | 544 |

三个长度的基础资料预填约 131 万 token；阶段间重跑前缀、开发集及延迟后缀重算均另外收费。代码不预先承诺 GPU 耗时。

阳性结果支持当前模型/语言/文本条件下的恢复可行性。继续研究前需要独立确认集、法语资料对照、第二种策略和方法差异复现。修复成功不证明自然模型只依赖循环状态；阴性不排除其他秩、头或恢复构造。

实现依据：[Qwen3.5 模型/配置](https://huggingface.co/Qwen/Qwen3.5-4B)、[固定 HF 实现](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[FLA](https://github.com/fla-org/flash-linear-attention)、[状态引导先例](https://arxiv.org/abs/2404.05971)、[混合模型通道干预](https://arxiv.org/abs/2609.04434)。

