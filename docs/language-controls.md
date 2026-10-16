# 下一轮最小生死实验：先排除语言倾向，再检查非语言规则

本轮只检验行为，不改循环状态、不改 KV、不插入提醒、不搜索干预参数。先回答：**固定资料与问题，模型还能否随 system 的要求切换输出？** 然后检验同样的控制损失是否出现在无自然语言选择的格式规则上。

默认模型仍为固定版本的 `Qwen/Qwen3.5-4B`，不是纯 Transformer 的 `Qwen3-4B`。两张 A100 80GB 各运行一份模型，按文档编号奇偶分工；同一文档的所有条件在同一张卡上运行，每次回答都从 token 0 独立预填。无训练、无 DDP、无 tensor parallel。两关最多 **480 条正式回答**；不增加调参集或独立 smoke 回答。每个 worker 仍先执行实际缓存路径检查，失败即停止。

**当前协议为 v2。** PR3 复审修复了混合语言背景漏检、生成截断造成的假阳性、复核文件静默回退和跨阶段续跑问题。请重新制备到下方 `*-v2` 目录；已生成的 v1 数据/结果保留原样，不能续接或混入 v2。模型、样本量、长度、提问因素及效应门槛不变，新增材料分块语言审核及零生成截断的有效性要求。

**当前不能直接开跑：默认语料不足。** 对固定 OPUS Books 全量源进行 v2 规则重筛，即使检查更短完整窗口并允许整对列方向校正，也只找到 **14 组**合格材料，少于登记的 16 组。默认 `controls-prepare` 现在会在下载/模型加载前明确拒绝运行。需要先批准扩充或更换平行语料并验证至少 16 组合格材料；不降低样本量、不放宽长度/纯度、不复用 v1 混合语言样本。下面的运行流程以这一数据阻断已解决为前提，不代表当前默认配置可用。

## 1. 固定设计

| 因素 | 语言关 A | 格式关 B |
|---|---|---|
| 配对材料 | 16 组新英法对齐片段及新随机记录 | 同一组材料及记录 |
| system 规则 | 英语 / 法语 / 不指定输出语言 | 严格 JSON / 严格 XML |
| 背景 | 无背景 / 英语约 8K / 同义法语约 8K | 同左 |
| 最终提问 | 英语 / 法语，询问相同两个字段 | 同左 |
| 正式回答数 | 16 × 3 × 3 × 2 = 288 | 16 × 2 × 3 × 2 = 192 |

共同的 system 文本用英文书写；语言关的三类 system 都要求两句完整回答，只有前两类额外指定输出语言。格式关不含“两句回答”或自然语言要求，仅指定结构。最终问题不重复语言/格式规则。记录始终为相同的管线表格，英文字段名固定，不随提问语言翻译；记录位于背景之后、问题之前。这一轮研究近期事实保留，不混入远距离事实检索。

JSON 必须恰好包含 `code`（字符串）和 `budget_eur`（整数）；XML 必须恰好为 `<answer>` 根下各一个 `<code>`、`<budget_eur>` 子元素。代码围栏、解释、额外字段、重复字段、XML 属性/声明等不合格。结构正确与事实正确分别计分。语言关的主规则指标是输出语言，不把句数误当成语言遵循指标。

固定配置见 [configs/controls.json](../configs/controls.json)：新记录 seed `26101717`，材料 seed `26101718`，贪心生成最多 128 token，非思考模板，预填 chunk 2048，SDPA + FLA，BF16 权重及缓存检查要求的 FP32 GDN 长状态。已有参数、长度和门槛不根据输出调整。

## 2. 英法材料如何配对

默认读取 [Helsinki-NLP/opus_books 的 en-fr 数据](https://huggingface.co/datasets/Helsinki-NLP/opus_books)，固定 revision `1f9f6191d0e91a3c539c2595e2fe48fc1420de9b`、Parquet 文件及其 SHA256。它提供逐单元英法对齐文本，**不是分别抽两批独立文章**。

程序沿源顺序收集完整对齐单元，要求两种实际文本都处于 **7783–8601 token**，组内长度差除以较长一边不超过 5%。不单独截断某种语言、不添加填充。检测到明显指令改写、特殊角色 token 或格式示例时清空窗口。两边须同时通过整段和 **256-token 分块**的语言检查，每块都须识别为对应语言且置信度 ≥0.98；不足 128 token 的尾块并入上一块，不漏检尾部。

数据源部分书籍的列语言方向反转：只有整对的语言识别均高置信度反向时，才交换**整对**原文列，并再次执行所有分块检查；保存 `source_columns` 记录，原文没有改写。片段内部混合语言仍不合格，不逐段拼凑/交换，不做机器翻译。每个位置除检查最长窗口外，也通过同时移除两边相同的完整对齐单元检查更短后缀，避免漏掉符合长度条件的窗口；仍不单侧截断。合格窗口不复用源单元 ID，全部扫描完成后固定 seed 抽取 16 组。条件不足直接报错，不自动放宽长度、换 seed 或改任务。

v1 的数据审查发现 3/16 组含连续大段异语文本（原 `controls-003`、`controls-011`、`controls-015`），虽然整段分类通过，也不能用于纯英/法背景对照。因此 v1 的“17 个合格窗口”不作为 v2 的有效性证据。全量 v2 筛选结果为 14 组，不足 16；未生成正式 v2 输入或模型回答。v2 保存完整分块检测记录并检查覆盖范围，仍需人工检查语义对齐。这些制备检查**不运行 4B 模型，也不产生实验结果**。

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

1. **没有生成截断。** 任一回答达到上限而未观察到结束 token，该关即标为 `*_generation_censored`，不能判为规则失守，也不能判为“只有语言效应”。各格截断数单独报告；此时数值损失仅为描述值，标记 `descriptive_only_censored`，不作机制或生死归因。不能删去这些回答或用复核标签将截断改为正常结束。若需增加预算，应另行登记完整实验，不挑样本补跑。
2. 每个短条件的“显式规则 × 提问语言”格均至少 **15/16 联合成功**。
3. 每个“规则 × 背景 × 提问语言”格至少 **15/16 内容正确**，语言关包括无语言规则条件。事实能力下降时不能归因于指令维护。
4. 文档平均控制损失 **≥25 个百分点**。
5. 至少 **4 个文档**的净控制损失为正。
6. 两种提问语言各自的平均控制损失均 **>0**。

A 过关才能启动 B。自动评分只决定是否暂时继续省算力，**不是最终科研结论**。如果 A 因自动误判停下，允许用绑定原答案的盲审标签纠正再启动 B；不允许改样本、长度、指标或门槛来救结果。若盲审推翻 A 的阳性，即便已跑 B 也不能宣称通过两关。

| 复核后结果 | 解释与行动 |
|---|---|
| 任一关出现生成截断 | 生成预算导致结果不完整，不能判断该关的指令维护效应 |
| A 短任务或事实保持不合格 | 任务无效，先查任务/评分，不能归因于维护失效 |
| A 有能力且保住事实，但未达效应门槛 | 本批控制下未复现足够强的语言规则损失，停止状态修复主线扩张 |
| A 通过、B 的能力或事实保持不合格 | 非语言关不能用于归因，不能称为语言特有或一般维护失效 |
| A 通过、B 有能力但未达效应门槛 | 目前只有语言特定证据，收窄故事；8K 的阴性不证明更长文本或其他规则稳定 |
| A、B 都通过 | 支持“本模型、本批材料、8K 下不只语言规则受影响”的行为主线，才值得下一步机制实验 |

即使两关阳性，也**没有判决 M1/M2，没有证明信息存于循环层，更没有证明已有恢复算子的新颖性或 ACL 可录用性**。

## 4. 服务器运行：两张 A100 80GB

**先解决上述语料阻断，再运行此节。** 以下示例假设已批准并验证一个足量的本地平行 JSONL，路径需替换为实际路径。在本 PR 分支、仓库根目录执行。可沿用之前经过检查的 `rcuf` 环境；新建环境的 CUDA/FLA 安装步骤见主 README。不要改 Grace 环境。

```bash
conda activate rcuf
git fetch origin experiment/language-controls
git switch --track origin/experiment/language-controls
python -m pip install -e '.[dev]'
python -m pytest
python scripts/doctor.py

python -m rcuf.cli controls-prepare \
  --parallel-corpus-jsonl /absolute/path/to/approved-en-fr.jsonl \
  --data-dir data/language-controls-v2

python -m rcuf.cli controls-launch \
  --data-dir data/language-controls-v2 \
  --run-dir runs/language-controls-v2 \
  --phase all --devices 0,1
```

`controls-prepare` 默认下载/复用固定 revision 的官方 checkpoint tokenizer；生成阶段下载/复用其权重。若服务器已有该 revision 的原始权重，可在制备命令追加 `--model /absolute/path/to/Qwen3.5-4B`。本地目录必须确实对应上述固定 revision，不能使用训练后权重；目录的 tokenizer/config 检查不等价于逐权重文件内容验证，程序不会替你证明本地副本的官方来源。没有可确认的本地副本时优先使用默认模型 ID。

`--phase all` 顺序跑 A、暂定门槛通过后再跑 B；A 未过会保存汇总并报错，**不会启动 B**。默认不指定 `--phase` 则只跑 A，适合先人工复核再决定 B。终端建议放在 tmux 中。启动器每约 30 秒报告 worker 状态，逐条进度在 `language/worker-*.log`、`format/worker-*.log`；一边失败会停止另一边。

断点续跑：不修改源码、准备数据、依赖环境和卡序，重跑同一条 launch 命令即可。已完成条件不会再生成；启动器在汇总任何阶段前，先持相应 worker/GPU 锁恢复各阶段不完整的最后一行，已完成行损坏会报错；仍存活的 worker 持锁时拒绝修改其文件。运行目录绑定源码、输入、运行时、GPU UUID 和逐行输出哈希；使用不同配置或不同物理 GPU 请另开运行目录。GPU 显存/耗时会记录，但本地未实测 A100 4B 性能，不承诺运行时长。

## 5. 盲审、阶段救回与结果交付

```bash
python -m rcuf.cli controls-summarize \
  --data-dir data/language-controls-v2 --run-dir runs/language-controls-v2

python -m rcuf.cli controls-review \
  --data-dir data/language-controls-v2 --run-dir runs/language-controls-v2 \
  --output runs/language-controls-v2/review-unlabeled.jsonl
```

导出文件随机顺序，隐藏 system 目标、背景/问题条件和自动分数，只展示答案与正确记录。请保留 `cell_id`、`output_sha256`、`answer`、`expected_target`，填写：

- `content_ok`: JSON 布尔值。两个字段均准确、肯定地给出且无矛盾/错误项目才为真；仅出现正确数字或代码不算。缺失字段、否定正确答案、附加错误金额均为假。
- `response_language`: `en` / `fr` / `mixed` / `other` / `unknown`。忽略项目 ID、代码、数字及 EUR；明显混合完整英文/法语片段用 `mixed`，没有足够自然语言证据用 `unknown`。纯 JSON/XML 可留空，语言关必须填。
- `reviewer_type`: `human` 或 `model`；`reviewer_id`: 实际复核者标识。LLM 复核必须标记 `model`，不能伪装人工。格式结构严格由解析器决定，不通过主观标签放宽；对合法 JSON/XML，内容标签必须与两个字段的精确比较一致，矛盾标签会报错。

`langid` 只提供初筛，短句/混合语言、事实否定都可能误判。正式结论要求本次**全部已生成答案**有人工标签；仅模型标签或部分人工标签时，`decision` 保持 `pending_human_review`。不要把自动 `provisional_interpretation` 当成最终结论。

显式提供 `--reviews` 时，文件不存在会立即报错，不能退回自动评分。summary 可展示部分复核的进度；但用复核标签启动 B 时必须完整填写全部 288 条 A 标签，不能只复核有利样本。文件哈希从实际参与评分的同一份内存快照计算，避免编辑过程中分数与标签指纹不一致。

将填写后的标签另存为 `review-filled.jsonl`：

```bash
python -m rcuf.cli controls-summarize \
  --data-dir data/language-controls-v2 --run-dir runs/language-controls-v2 \
  --reviews runs/language-controls-v2/review-filled.jsonl
```

如果只跑过 A，且人工复核后的 A 达标，可继续：

```bash
python -m rcuf.cli controls-launch \
  --data-dir data/language-controls-v2 --run-dir runs/language-controls-v2 \
  --phase format --devices 0,1 \
  --reviews runs/language-controls-v2/review-filled.jsonl
```

B 的首次启动把门槛来源与复核文件哈希写入 `format-gate.json`。续跑 B 必须使用相同参数和**未修改的该文件**。全部生成后，用新的 `--output` 路径导出 480 行复核表；已有 A 标签可按 `cell_id` 转移，保持答案/哈希/目标绑定，补齐 B 内容标签，再用新的完整复核文件做最终 summarize。原始复核文件不可覆盖，也不要修改已经用于 B 续跑的标签文件。

最终交回 `controls-summary.json`、`manifest.json`、两阶段的 `results-*.jsonl`、日志、环境/缓存检查、`format-gate.json`（如启动过 B）及完整人工/模型复核表。不要只提供汇总比例。生成的完整背景文本和输入仅供本地核验，不随结果公开分发。
