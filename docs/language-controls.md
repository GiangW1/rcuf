# 下一轮最小生死实验：先排除语言倾向，再检查非语言规则

本轮只检验行为，不改循环状态、不改 KV、不插入提醒、不搜索干预参数。先回答：**固定资料与问题，模型还能否随 system 的要求切换输出？** 然后检验同样的控制损失是否出现在无自然语言选择的格式规则上。

默认模型仍为固定版本的 `Qwen/Qwen3.5-4B`，不是纯 Transformer 的 `Qwen3-4B`。两张 A100 80GB 各运行一份模型，按文档编号奇偶分工；同一文档的所有条件在同一张卡上运行，每次回答都从 token 0 独立预填。无训练、无 DDP、无 tensor parallel。两关最多 **480 条正式回答**；不增加调参集或独立 smoke 回答。每个 worker 仍先执行实际缓存路径检查，失败即停止。

## 1. 固定设计

| 因素 | 语言关 A | 格式关 B |
|---|---|---|
| 独立材料 | 16 组新英法对齐片段及新随机记录 | 同一组材料及记录 |
| system 规则 | 英语 / 法语 / 不指定输出语言 | 严格 JSON / 严格 XML |
| 背景 | 无背景 / 英语约 8K / 同义法语约 8K | 同左 |
| 最终提问 | 英语 / 法语，询问相同两个字段 | 同左 |
| 正式回答数 | 16 × 3 × 3 × 2 = 288 | 16 × 2 × 3 × 2 = 192 |

共同的 system 文本用英文书写；语言关的三类 system 都要求两句完整回答，只有前两类额外指定输出语言。格式关不含“两句回答”或自然语言要求，仅指定结构。最终问题不重复语言/格式规则。记录始终为相同的管线表格，英文字段名固定，不随提问语言翻译；记录位于背景之后、问题之前。这一轮研究近期事实保留，不混入远距离事实检索。

JSON 必须恰好包含 `code`（字符串）和 `budget_eur`（整数）；XML 必须恰好为 `<answer>` 根下各一个 `<code>`、`<budget_eur>` 子元素。代码围栏、解释、额外字段、重复字段、XML 属性/声明等不合格。结构正确与事实正确分别计分。语言关的主规则指标是输出语言，不把句数误当成语言遵循指标。

固定配置见 [configs/controls.json](../configs/controls.json)：新记录 seed `26101717`，材料 seed `26101718`，贪心生成最多 128 token，非思考模板，预填 chunk 2048，SDPA + FLA，BF16 权重及缓存检查要求的 FP32 GDN 长状态。已有参数、长度和门槛不根据输出调整。

## 2. 英法材料如何配对

默认读取 [Helsinki-NLP/opus_books 的 en-fr 数据](https://huggingface.co/datasets/Helsinki-NLP/opus_books)，固定 revision `1f9f6191d0e91a3c539c2595e2fe48fc1420de9b`、Parquet 文件及其 SHA256。它提供逐单元英法对齐文本，**不是分别抽两批独立文章**。

程序沿源顺序收集完整对齐单元，要求两种实际文本都处于 **7783–8601 token**，组内长度差除以较长一边不超过 5%。不单独截断某种语言、不添加填充。检测到明显指令改写、特殊角色 token 或格式示例时清空窗口；两边整段语言识别置信度须 ≥0.98。合格窗口不复用源单元 ID，全部扫描完成后固定 seed 抽取 16 组。条件不足直接报错，不自动放宽长度、换 seed 或改任务。

本地用正式固定 tokenizer 已完成一次完整制备：找到 17 个合格窗口，选出 16 组并生成全部 480 个官方模板输入。此检查**没有运行 4B 模型，也没有产生实验结果**。

注意三个限制：

- OPUS Books 是翻译文学文本，不是上一轮 Wikipedia 资料；本轮不是对上一轮材料的精确复现。数据卡也提示部分对齐不完善，自动筛选不等于人工核验。运行前请检查 `backgrounds.jsonl` 的英法对应及是否残留指令性文本。
- 不重叠源单元不等于不同书籍。多个片段可能来自同一本书；16 个文档的区间只反映本批片段，不提供跨书籍/跨语料的独立性保证。翻译风格、英文 system 和固定英文字段名也限制外推。
- 原始文本受数据源许可与使用条件约束，适用研究/教育等用途；不将全文材料、准备后的 token 流或模型权重提交进 Git。源码只固定来源与抽样算法。

也可在**看输出之前**指定自有对齐 JSONL，每行例如 `{"id":"unique-source-unit","translation":{"en":"...","fr":"..."}}`，使用 `--parallel-corpus-jsonl /path/to/en-fr.jsonl`。必须提供足够多、可连续拼接的真实对齐单元，源 ID 唯一；一整篇不满足长度约束的文本不会被单侧截断。源文件 SHA256 写入 manifest。换源构成新实验，不可沿用默认语料的结论或混入旧运行目录。

## 3. 主指标和固定生死门槛

对一个固定文档、背景和提问语言，只有**两个 system 条件都成功**才记一次切换成功：语言关须可分别输出英语和法语；格式关须可分别输出 JSON 和 XML。固定输出某一种语言或某一种格式不可能得到切换成功。

每个文档的控制损失为四个配对差值的平均：

```text
L_doc = mean over background∈{en,fr}, question∈{en,fr}
        [switch(short, question) − switch(background, question)]
```

主结果是 16 个 `L_doc` 的均值，并以**文档为单位**做 10,000 次 bootstrap；不把两种提问或多个 system 条件当独立样本。规则切换为主指标，规则+内容的联合切换并列报告。另导出全部分格结果、按提问语言的损失和相对于“无语言规则”的净语言增益，便于检查单向英语优势、背景吸引和问题语言效应。bootstrap 区间描述不确定性，以下门槛是筛选规则而非显著性检验。

每关必须同时满足：

1. 每个短条件的“显式规则 × 提问语言”格均至少 **15/16 联合成功**。
2. 每个“规则 × 背景 × 提问语言”格至少 **15/16 内容正确**，语言关包括无语言规则条件。事实能力下降时不能归因于指令维护。
3. 文档平均控制损失 **≥25 个百分点**。
4. 至少 **4 个文档**的净控制损失为正。
5. 两种提问语言各自的平均控制损失均 **>0**。

A 过关才能启动 B。自动评分只决定是否暂时继续省算力，**不是最终科研结论**。如果 A 因自动误判停下，允许用绑定原答案的盲审标签纠正再启动 B；不允许改样本、长度、指标或门槛来救结果。若盲审推翻 A 的阳性，即便已跑 B 也不能宣称通过两关。

| 复核后结果 | 解释与行动 |
|---|---|
| A 短任务或事实保持不合格 | 任务无效，先查任务/评分，不能归因于维护失效 |
| A 有能力且保住事实，但未达效应门槛 | 本批控制下未复现足够强的语言规则损失，停止状态修复主线扩张 |
| A 通过、B 的能力或事实保持不合格 | 非语言关不能用于归因，不能称为语言特有或一般维护失效 |
| A 通过、B 有能力但未达效应门槛 | 目前只有语言特定证据，收窄故事；8K 的阴性不证明更长文本或其他规则稳定 |
| A、B 都通过 | 支持“本模型、本批材料、8K 下不只语言规则受影响”的行为主线，才值得下一步机制实验 |

即使两关阳性，也**没有判决 M1/M2，没有证明信息存于循环层，更没有证明已有恢复算子的新颖性或 ACL 可录用性**。

## 4. 服务器运行：两张 A100 80GB

在本 PR 分支、仓库根目录执行。可沿用之前经过检查的 `rcuf` 环境；新建环境的 CUDA/FLA 安装步骤见主 README。不要改 Grace 环境。

```bash
conda activate rcuf
git fetch origin experiment/language-controls
git switch --track origin/experiment/language-controls
python -m pip install -e '.[dev]'
python -m pytest
python scripts/doctor.py

python -m rcuf.cli controls-prepare \
  --data-dir data/language-controls-v1

python -m rcuf.cli controls-launch \
  --data-dir data/language-controls-v1 \
  --run-dir runs/language-controls-v1 \
  --phase all --devices 0,1
```

`controls-prepare` 默认下载/复用固定 revision 的官方 checkpoint tokenizer；生成阶段下载/复用其权重。若服务器已有该 revision 的原始权重，可在制备命令追加 `--model /absolute/path/to/Qwen3.5-4B`。本地目录必须确实对应上述固定 revision，不能使用训练后权重；目录的 tokenizer/config 检查不等价于逐权重文件内容验证，程序不会替你证明本地副本的官方来源。没有可确认的本地副本时优先使用默认模型 ID。

`--phase all` 顺序跑 A、暂定门槛通过后再跑 B；A 未过会保存汇总并报错，**不会启动 B**。默认不指定 `--phase` 则只跑 A，适合先人工复核再决定 B。终端建议放在 tmux 中。启动器每约 30 秒报告 worker 状态，逐条进度在 `language/worker-*.log`、`format/worker-*.log`；一边失败会停止另一边。

断点续跑：不修改源码、准备数据、依赖环境和卡序，重跑同一条 launch 命令即可。已完成条件不会再生成；不完整的最后一行可恢复，已完成行损坏会报错。运行目录绑定源码、输入、运行时、GPU UUID 和逐行输出哈希；使用不同配置或不同物理 GPU 请另开运行目录。GPU 显存/耗时会记录，但本地未实测 A100 4B 性能，不承诺运行时长。

## 5. 盲审、阶段救回与结果交付

```bash
python -m rcuf.cli controls-summarize \
  --data-dir data/language-controls-v1 --run-dir runs/language-controls-v1

python -m rcuf.cli controls-review \
  --data-dir data/language-controls-v1 --run-dir runs/language-controls-v1 \
  --output runs/language-controls-v1/review-unlabeled.jsonl
```

导出文件随机顺序，隐藏 system 目标、背景/问题条件和自动分数，只展示答案与正确记录。请保留 `cell_id`、`output_sha256`、`answer`、`expected_target`，填写：

- `content_ok`: JSON 布尔值。两个字段均准确、肯定地给出且无矛盾/错误项目才为真；仅出现正确数字或代码不算。缺失字段、否定正确答案、附加错误金额均为假。
- `response_language`: `en` / `fr` / `mixed` / `other` / `unknown`。忽略项目 ID、代码、数字及 EUR；明显混合完整英文/法语片段用 `mixed`，没有足够自然语言证据用 `unknown`。纯 JSON/XML 可留空，语言关必须填。
- `reviewer_type`: `human` 或 `model`；`reviewer_id`: 实际复核者标识。LLM 复核必须标记 `model`，不能伪装人工。格式结构严格由解析器决定，不通过主观标签放宽。

`langid` 只提供初筛，短句/混合语言、事实否定都可能误判。正式结论要求本次**全部已生成答案**有人工标签；仅模型标签或部分人工标签时，`decision` 保持 `pending_human_review`。不要把自动 `provisional_interpretation` 当成最终结论。

将填写后的标签另存为 `review-filled.jsonl`：

```bash
python -m rcuf.cli controls-summarize \
  --data-dir data/language-controls-v1 --run-dir runs/language-controls-v1 \
  --reviews runs/language-controls-v1/review-filled.jsonl
```

如果只跑过 A，且人工复核后的 A 达标，可继续：

```bash
python -m rcuf.cli controls-launch \
  --data-dir data/language-controls-v1 --run-dir runs/language-controls-v1 \
  --phase format --devices 0,1 \
  --reviews runs/language-controls-v1/review-filled.jsonl
```

B 的首次启动把门槛来源与复核文件哈希写入 `format-gate.json`。续跑 B 必须使用相同参数和**未修改的该文件**。全部生成后，用新的 `--output` 路径导出 480 行复核表；已有 A 标签可按 `cell_id` 转移，保持答案/哈希/目标绑定，补齐 B 内容标签，再用新的完整复核文件做最终 summarize。原始复核文件不可覆盖，也不要修改已经用于 B 续跑的标签文件。

最终交回 `controls-summary.json`、`manifest.json`、两阶段的 `results-*.jsonl`、日志、环境/缓存检查、`format-gate.json`（如启动过 B）及完整人工/模型复核表。不要只提供汇总比例。生成的完整背景文本和输入仅供本地核验，不随结果公开分发。
