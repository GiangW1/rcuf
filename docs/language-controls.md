# 下一轮最小生死实验：先排除语言倾向，再检查非语言规则

本轮只检验行为，不改循环状态、不改 KV、不插入提醒、不搜索干预参数。先回答：**固定资料与问题，模型还能否随 system 的要求切换输出？** 然后检验同样的控制损失是否出现在无自然语言选择的格式规则上。

默认模型仍为固定版本的 `Qwen/Qwen3.5-4B`，不是纯 Transformer 的 `Qwen3-4B`。两张 A100 80GB 各运行一份模型，按文档编号奇偶分工；同一文档的所有条件在同一张卡上运行，每次回答都从 token 0 独立预填。无训练、无 DDP、无 tensor parallel。两关最多 **480 条正式回答**；不增加调参集或独立 smoke 回答。每个 worker 仍先执行实际缓存路径检查，失败即停止。

**当前协议为 v2，默认语料改为 OpenSubtitles 平行句对。** PR3 复审修复了混合语言背景漏检、生成截断造成的假阳性、复核文件静默回退和跨阶段续跑问题。请重新制备到下方 `*-v2-subtitles` 目录；已生成的 v1 或旧语料 v2 数据/结果保留原样，不能续接或混入。模型、16 组样本、长度、提问因素及所有效应/有效性门槛不变。

数据制备与独立重放审计均使用固定版本的真实 Qwen tokenizer，在 CPU 上执行，不运行 4B 生成模型。审计记录见 [data-audit-controls-v2-subtitles.json](data-audit-controls-v2-subtitles.json)；它是输入可用性的证据，**不是实验阳性结果或 A100 性能测试**。

本次全量扫描 3,870,286 行，得到 **24 个**满足现有全部自动条件的不重叠窗口；固定 seed 抽取 **16 组**，生成并重放验证 **480 条**完整输入（A 288、B 192）。32 段背景实际为 **7,783–8,205 token**，无需降低样本量或放宽长度条件。

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

默认读取 [Sentence Transformers 发布的 OpenSubtitles 英法句对](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-opensubtitles)，固定 revision `d86a387587ab6f2fd9ec7453b2765cec68111c87`，只使用 `en-fr/train-00000-of-00002.parquet`，**不是全库或两批独立文章**。该文件有 3,870,286 行、201,725,376 字节（约 202 MB）；`english` / `non_english` 分别映射到 en / fr，原始行序不变，行号作为源 ID。文件 SHA256 固定为 `9c1fc17a1d82ff60e0a243679c92c86967cd73eacaef6d8469915f42e0fef921`，下载后先校验内容再解析。

程序沿源顺序收集完整对齐单元，要求两种实际文本都处于 **7783–8601 token**，组内长度差除以较长一边不超过 5%。不单独截断某种语言、不添加填充。检测到明显指令改写、特殊角色 token 或格式示例时清空窗口。两边须同时通过整段和 **256-token 分块**的语言检查，每块都须识别为对应语言且置信度 ≥0.98；不足 128 token 的尾块并入上一块，不漏检尾部。

只有整对的语言识别均高置信度反向时，才允许交换**整对**原文列，并再次执行所有分块检查；保存 `source_columns` 记录，原文没有改写。片段内部混合语言仍不合格，不逐段拼凑/交换，不做机器翻译。每个位置除检查最长窗口外，也通过同时移除两边相同的完整对齐单元检查更短后缀，避免漏掉符合长度条件的窗口；仍不单侧截断。合格窗口不复用源单元 ID，**完整扫描上述登记文件之后**才按固定 seed 抽取 16 组，不是凑到 16 组就停止。条件不足直接报错，不自动放宽长度、换 seed 或改任务。

此前 OPUS Books 全量严格筛选只剩 14 组，不足 16；用户已同意扩充/更换语料。本次换源没有放宽任何门槛，也没有把旧 Books 材料拼入新库。v1 中原 `controls-003`、`controls-011`、`controls-015` 的大段异语问题仍视为无效，不能复用。另检查的 Europarl 第一分片（1,025,507 行）没有找到满足全部条件的窗口，也未进入正式数据。

独立审计会重新读取固定源，逐一重建选中片段，核对源单元/片段哈希、实际 token 数、整段及全部语言分块，并从完整聊天模板重放 480 条输入，核对 token、问题切分点及指纹。审计不替代语义对齐人工检查；完整候选池数量由制备时全文件扫描得到，独立审计只重检选中片段。不要把这些输入检查称为模型结果。

另由代码审查代理抽看每组选中窗口的起点、1/4、中点、3/4、末尾，共 80 个源单元。多数译文情节/人物对应，但可见个别意译或单句错配；这只是代理抽查，不是人工完成的 16 组全文逐句双语认证。多个片段也出现 Darrin / Endora 等重复角色名，不应视为 16 部独立作品。不要用语言识别置信度或输入重放成功来代替这些语义与独立性限制。

注意三个限制：

- 背景现在是影视字幕句对形成的片段，不是 Wikipedia、文学或议会文本；句对的源文件行序不等于完整连贯文章。发布方数据卡明确提醒存在空白和译文错配，语言检测通过不代表语义等同。结果只能解释本批筛选后的字幕背景，不能声称精确复现上一轮 Wikipedia 效应。运行前请检查 `backgrounds.jsonl` 的英法对应及是否残留指令性文本。
- 不重叠源行不等于不同影视作品。相同作品的字幕版本、重复台词和说话风格可能相关；16 个片段的区间不提供跨作品/跨语料的独立性保证。严格等长筛选也可能偏向特定翻译风格，英文 system 和固定英文字段名同样限制外推。
- 原始文本受数据源许可与使用条件约束；数据发布页未提供明确统一许可，本仓库不授予语料再分发权。遵守 [OpenSubtitles](https://www.opensubtitles.org/) 及 [OPUS 原始语料说明](https://opus.nlpl.eu/OpenSubtitles.php) 的使用/署名要求；论文引用 Lison & Tiedemann (2016)。不将全文材料、准备后的 token 流或模型权重提交进 Git，只公开来源、算法及无正文的审计元数据。

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

在本 PR 分支、仓库根目录执行。可沿用之前经过检查的 `rcuf` 环境；新建环境的 CUDA/FLA 安装步骤见主 README。不要改 Grace 环境。先保存现有本地修改，不要丢弃旧实验。下载 tokenizer/语料及 CPU 筛选无需占用 GPU；制备会扫描完整登记文件，请等待最终输出 16 组和 480 条输入。

```bash
conda activate rcuf
git fetch origin experiment/language-controls
git switch experiment/language-controls
git merge --ff-only origin/experiment/language-controls
python -m pip install -e '.[dev]'
python -m pytest
python scripts/doctor.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m rcuf.cli controls-prepare \
  --data-dir data/language-controls-v2-subtitles

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.audit_controls_data \
  --data-dir data/language-controls-v2-subtitles \
  --output data/language-controls-v2-subtitles/data-audit.json

python -m rcuf.cli controls-launch \
  --data-dir data/language-controls-v2-subtitles \
  --run-dir runs/language-controls-v2-subtitles \
  --phase all --devices 0,1
```

`controls-prepare` 默认下载/复用固定 revision 的官方 checkpoint tokenizer；生成阶段下载/复用其权重。若服务器已有该 revision 的原始权重，可在制备命令追加 `--model /absolute/path/to/Qwen3.5-4B`。本地目录必须确实对应上述固定 revision，不能使用训练后权重；目录的 tokenizer/config 检查不等价于逐权重文件内容验证，程序不会替你证明本地副本的官方来源。没有可确认的本地副本时优先使用默认模型 ID。

全量 CPU 筛选可能需要数十分钟，每扫描 10,000 行会打印进度；此时没有加载 4B 权重或占用 GPU，不要因一段时间没增加合格数量而放宽条件。若 Hugging Face 的 Xet 下载卡住，可只在制备命令前追加 `HF_HUB_DISABLE_XET=1` 重试；保持数据源、revision、文件哈希和配置不变。独立审计输出文件不能覆盖，重做审计请另用新文件名。

`--phase all` 顺序跑 A、暂定门槛通过后再跑 B；A 未过会保存汇总并报错，**不会启动 B**。默认不指定 `--phase` 则只跑 A，适合先人工复核再决定 B。终端建议放在 tmux 中。启动器每约 30 秒报告 worker 状态，逐条进度在 `language/worker-*.log`、`format/worker-*.log`；一边失败会停止另一边。

断点续跑：不修改源码、准备数据、依赖环境和卡序，重跑同一条 launch 命令即可。已完成条件不会再生成；启动器在汇总任何阶段前，先持相应 worker/GPU 锁恢复各阶段不完整的最后一行，已完成行损坏会报错；仍存活的 worker 持锁时拒绝修改其文件。运行目录绑定源码、输入、运行时、GPU UUID 和逐行输出哈希；使用不同配置或不同物理 GPU 请另开运行目录。GPU 显存/耗时会记录，但本地未实测 A100 4B 性能，不承诺运行时长。

## 5. 盲审、阶段救回与结果交付

```bash
python -m rcuf.cli controls-summarize \
  --data-dir data/language-controls-v2-subtitles --run-dir runs/language-controls-v2-subtitles

python -m rcuf.cli controls-review \
  --data-dir data/language-controls-v2-subtitles --run-dir runs/language-controls-v2-subtitles \
  --output runs/language-controls-v2-subtitles/review-unlabeled.jsonl
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
  --data-dir data/language-controls-v2-subtitles --run-dir runs/language-controls-v2-subtitles \
  --reviews runs/language-controls-v2-subtitles/review-filled.jsonl
```

如果只跑过 A，且人工复核后的 A 达标，可继续：

```bash
python -m rcuf.cli controls-launch \
  --data-dir data/language-controls-v2-subtitles --run-dir runs/language-controls-v2-subtitles \
  --phase format --devices 0,1 \
  --reviews runs/language-controls-v2-subtitles/review-filled.jsonl
```

B 的首次启动把门槛来源与复核文件哈希写入 `format-gate.json`。续跑 B 必须使用相同参数和**未修改的该文件**。全部生成后，用新的 `--output` 路径导出 480 行复核表；已有 A 标签可按 `cell_id` 转移，保持答案/哈希/目标绑定，补齐 B 内容标签，再用新的完整复核文件做最终 summarize。原始复核文件不可覆盖，也不要修改已经用于 B 续跑的标签文件。

最终交回 `controls-summary.json`、`manifest.json`、两阶段的 `results-*.jsonl`、日志、环境/缓存检查、`format-gate.json`（如启动过 B）及完整人工/模型复核表。不要只提供汇总比例。生成的完整背景文本和输入仅供本地核验，不随结果公开分发。
