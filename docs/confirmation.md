# PR #1 后的最小确认实验

本轮只确认三个问题：原样本上“相加比替换持久”的差异能否跨卡复现；独立新样本上漂移从哪些已测长度开始出现、法文资料下是否也出现；匹配状态改动幅度后，方法差异是否仍在，文字提醒在 4096-token 延迟后表现如何。

模型固定为 Qwen3.5-4B，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；rank=4，原始相加 γ=1、原始替换 λ=1。沿用上一轮开发集选出的参数，本轮没有开发集、搜索网格或基于新测试结果的选择步骤。用两张 A100 80GB 或两张 RTX 5090，每张卡独立加载模型，按文档分工。

## 实验规模与配对

| 部分 | 输入与条件 | 回答数 |
|---|---|---:|
| 交叉 A | 原 32 个 32K 测试输入；延迟 4096；baseline / add_lowrank / replace | 96 |
| 交叉 B | 完全相同输入与方法，每份文档交换到另一张卡 | 96 |
| 新样本长度与背景 | 32 组新记录；0 / 512 / 2048 / 8192 / 32768 token；英文与法文资料 | 288 |
| 新样本方法确认 | 英文 32K；即时与延迟 4096；5 个条件，复用上表 32 条 baseline | 新增 288 |
| 合计 | 768 条正式回答，另有 6 条短 smoke | 774 |

5 个条件为未干预、低秩相加、原始投影替换、等幅度投影方向、文字提醒。所有同一文档的方法都在同一卡运行；交叉 B 才交换文档与 GPU 的对应关系。启动时两次均传相同顺序的 `--devices`，程序负责交换文档。

新记录用固定的新种子生成，与原轮次问答分离。新资料按文章文本哈希排除原轮次全部开发与测试原文，各新文档之间也不复用相同原文。每组新记录在所有长度和背景下完全相同；0 token 的公共条件只生成一次。资料来自固定 revision 的英文/法文 Wikipedia，制备时检查资料语言。这是背景语料对照，**不是逐句对齐的翻译，不能单独把差异归因为语言**。

每个干预位置都从 token 0 独立预填，chunk size=2048；该位置的各方法从同一未改动快照分叉。恢复中断运行不会改变已有条件的预填切块。文字提醒在对应位置插入原提示的 token 流，额外 token 数、完整实际输入哈希都会记录；它可能位于 `<reference>` 内，其结果只代表这种插入方式。

## 等幅度条件的定义

对每个循环层分别计算未干预状态上的两个方向：

```
delta_add = P (S_F - S_E)
delta_replace = P (S_F - S_current)
delta_matched = delta_replace * ||delta_add||_F / ||delta_replace||_F
```

范数在该层的所有头和矩阵元素上汇总。每层匹配，因此全模型总改变量也匹配。仅改变循环长状态，KV、卷积缓存和序列位置保持不变。每层目标范数与实际范数写入结果，运行时检查误差。

该条件检验等幅度下的方向差异；它是软替换，**不具有原始 λ=1 投影替换的幂等性**。如果方向为零但目标范数非零，直接报错，避免悄悄跳过。

## 在服务器运行

先切到本 PR 分支，进入仓库并激活原来的 Python 3.11 实验环境。保持原轮次依赖：PyTorch 2.8.0、Transformers 5.19.0、FLA 0.5.2。A100 沿用原 A100 环境；5090 使用支持该卡的 CUDA wheel。程序会记录实际版本、GPU UUID、模型实现和输入指纹。

```bash
python -m pip install -e '.[dev]'

python -m rcuf.cli confirm-prepare \
  --pilot-data-dir /mnt/mydata/wja/rcuf/data/pilot-20261016 \
  --data-dir data/confirmation

python -m rcuf.cli confirm-launch \
  --data-dir data/confirmation \
  --run-dir runs/confirmation \
  --devices 0,1

python -m rcuf.cli confirm-summarize \
  --data-dir data/confirmation \
  --run-dir runs/confirmation

python -m rcuf.cli confirm-review \
  --data-dir data/confirmation \
  --run-dir runs/confirmation \
  --output runs/confirmation/review-unlabeled.jsonl
```

`confirm-prepare` 需要原服务器的真实 `manifest.json` 和 `examples.jsonl`；报告中的 manifest 不能代替输入 token。它核对 PR #1 的原始数据指纹，不会重造旧输入。如果原模型已搬家，加 `--model /新的/Qwen3.5-4B/目录`；也可用 `--model Qwen/Qwen3.5-4B` 从固定 revision 加载。

可在制备时加 `--english-corpus-jsonl /path/en.jsonl` 与 `--french-corpus-jsonl /path/fr.jsonl`，避免在线流式下载；每行至少有 `text`，可附 `title`、`url`。原英文导出文件可继续作为来源，程序会跳过已用文章。来源文件 SHA256 会冻结进 manifest。不传时从固定 Wikipedia revision 流式读取。

默认依次运行 smoke、交叉 A、交叉 B、新样本确认。也可用 `--phase smoke`、`--phase crossover-a`、`--phase crossover-b`、`--phase confirm` 单独运行。中断后重复相同 `confirm-launch` 命令即可续跑，**不要重新执行 prepare 或修改代码/数据/卡的顺序**。每阶段两个 worker 的日志都在 `runs/confirmation/<phase>/worker-0.log` 和 `worker-1.log`。跨轮次检测到 GPU UUID 换位、运行指纹不同或重复物理 GPU 时会拒绝比较。

## 复核与预先固定的判断

汇总文件为 `runs/confirmation/confirmation-summary.json`。生成结束即可查看自动评分表，正式判定要求 768 条回答全部完成复核。导出隐藏方法、长度、背景和 GPU 条件；填写 `language_ok`、`content_ok`、`reviewer_type`（`human` 或 `model`）及 `reviewer_id`，保持答案、预期字段、cell ID 和输出哈希原样。

```bash
python -m rcuf.cli confirm-summarize \
  --data-dir data/confirmation \
  --run-dir runs/confirmation \
  --reviews runs/confirmation/review-filled.jsonl
```

法语判断针对完整回答的自然语言，项目名/代码/金额不算其他语言；内容判断要求肯定地给出正确项目对应的代码与预算，不能有否定、矛盾或错误归属。仅匹配答案中的字符串不足以判为内容正确。

模型复核可产生 `provisional_decision`，最终 `decision` 保持 `pending_human_review`；全部明确标注为人工复核后才给出最终确认状态。复核文件出现改答案、改预期目标、重复标签或矛盾来源时会报错。

主要确认条件全部预先固定：

1. 原样本每种方法在两张卡之间的联合成功标签不一致最多 1/32；完整输出 token 的差异单独报告。
2. 两次原样本重跑中，相加相对原始替换的联合成功净优势各至少 7/32。
3. 新样本短条件语言、内容各至少 31/32；英文 32K 语言遵循相对短条件净下降至少 7/32，内容仍至少 29/32。
4. 新样本延迟 4096 的相加联合成功至少 29/32，比原始替换净高至少 7/32；其内容相对同位置未干预对照净损失最多 1/32。

满足全部条件才将原始现象及方法差异标为 `confirmed`。否则为 `not_confirmed`，汇总保留各条件结果，不修改阈值。这里确认的是上一轮模式，不代表新方法已经优于现有方法。

等幅度条件、文字提醒和法文资料分别给出配对结果：如果等幅度下相加仍净高至少 7/32，才支持继续研究方向差异；否则不能沿用原始幅度下的优势结论。若两卡联合标签不稳定，先查数值与运行路径。若新样本不复现原始差异，应暂停扩展该机制解释。

512/2048/8192 等点只给出“已测点中首次大幅下降的位置”，不能当作精确寿命或半衰期。配对 bootstrap 使用文档为单位；32/32 的经验 bootstrap 可能退化，不能解读为总体成功率确定为 100%。状态干预可恢复行为也不证明自然模型只依赖循环状态。

## 本地与服务器验证

```bash
python -m pytest
python -m ruff check rcuf tests scripts
```

测试包含真实小型 HF 模型的缓存分叉、干预后续接、KV/卷积保持和逐层范数匹配；合成结果仅用于验证配对、完整性与判定代码。4B 模型的实际确认结果需要运行上述服务器命令后获得。
