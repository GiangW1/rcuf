# RCUF 两卡先导实验

**审阅状态：模型语义复核已记录，正式人工盲审尚未完成。下列结果用于探索，不作为经过人工盲审的最终结论。**

运行状态：`pilot_complete_model_reviewed`。All pilot stages complete with model semantic audits; final human blind review remains pending. Delay gate: stop_current_configuration

32K 条件：未干预联合成功 0/32，投影恢复 32/32。低秩相加 32/32，文字提醒 32/32；本先导尚未显示投影恢复相对这些方法的额外优势。

延迟 512 token：投影恢复联合成功 31/32，低秩相加 32/32。

延迟 4096 token：投影恢复联合成功 11/32，低秩相加 32/32。

4096 token 后投影恢复仅保留即时净收益的 34.4%，未达到预设至少一半的门槛；当前配置停止继续投入。

本轮共生成 768 条独立回答（开发集与测试集，不含 smoke）；另外复用 64 条已复核基线。

模型：Qwen3.5-4B，BF16，Transformers 5.19.0，FLA 0.5.2，PyTorch 2.8.0+cu128。

规模：8 篇开发文档、32 篇独立测试文档；参考资料长度 0/8192/32768 token；rank=4。

baseline 使用物理 GPU 6、7；后续阶段使用物理 GPU 5、6（RTX 5090，每卡独立加载模型，按文档分片）。

开发集冻结强度：`{"add_full": 1.0, "add_lowrank": 1.0, "replace": 1.0}`。

## baseline

结果行数：96；完整：True；模型复核完整：True；判定：`continue_pilot`。

| 方法 | 强度 | 参考长度 | 延迟 | 法语遵循 | 内容正确 | 联合成功 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 0 | 0 | 32/32 | 32/32 | 32/32 |
| baseline | 0.0 | 8192 | 0 | 1/32 | 32/32 | 1/32 |
| baseline | 0.0 | 32768 | 0 | 0/32 | 32/32 | 0/32 |

门槛：`{"short_language_capable": true, "short_content_capable": true, "large_natural_drift": true, "long_content_preserved": true}`。

## tune

结果行数：160；完整：True；模型复核完整：True；判定：`ready`。

| 方法 | 强度 | 参考长度 | 延迟 | 法语遵循 | 内容正确 | 联合成功 |
|---|---:|---:|---:|---:|---:|---:|
| add_full | 0.25 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| add_full | 0.25 | 32768 | 0 | 0/8 | 8/8 | 0/8 |
| add_full | 1.0 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| add_full | 1.0 | 32768 | 0 | 8/8 | 8/8 | 8/8 |
| add_full | 4.0 | 0 | 0 | 8/8 | 7/8 | 7/8 |
| add_full | 4.0 | 32768 | 0 | 8/8 | 8/8 | 8/8 |
| add_lowrank | 0.25 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| add_lowrank | 0.25 | 32768 | 0 | 0/8 | 8/8 | 0/8 |
| add_lowrank | 1.0 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| add_lowrank | 1.0 | 32768 | 0 | 8/8 | 8/8 | 8/8 |
| add_lowrank | 4.0 | 0 | 0 | 8/8 | 7/8 | 7/8 |
| add_lowrank | 4.0 | 32768 | 0 | 8/8 | 8/8 | 8/8 |
| baseline | 0.0 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| baseline | 0.0 | 32768 | 0 | 0/8 | 8/8 | 0/8 |
| replace | 0.25 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| replace | 0.25 | 32768 | 0 | 0/8 | 8/8 | 0/8 |
| replace | 0.5 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| replace | 0.5 | 32768 | 0 | 7/8 | 8/8 | 7/8 |
| replace | 1.0 | 0 | 0 | 8/8 | 8/8 | 8/8 |
| replace | 1.0 | 32768 | 0 | 8/8 | 8/8 | 8/8 |

## compare

结果行数：384；完整：True；模型复核完整：True；判定：`continue_pilot`。

| 方法 | 强度 | 参考长度 | 延迟 | 法语遵循 | 内容正确 | 联合成功 |
|---|---:|---:|---:|---:|---:|---:|
| add_full | 1.0 | 0 | 0 | 32/32 | 32/32 | 32/32 |
| add_full | 1.0 | 32768 | 0 | 32/32 | 32/32 | 32/32 |
| add_lowrank | 1.0 | 0 | 0 | 32/32 | 32/32 | 32/32 |
| add_lowrank | 1.0 | 32768 | 0 | 32/32 | 32/32 | 32/32 |
| baseline | 0.0 | 0 | 0 | 32/32 | 32/32 | 32/32 |
| baseline | 0.0 | 32768 | 0 | 0/32 | 32/32 | 0/32 |
| reminder | 0.0 | 0 | 0 | 32/32 | 32/32 | 32/32 |
| reminder | 0.0 | 32768 | 0 | 32/32 | 32/32 | 32/32 |
| replace | 1.0 | 0 | 0 | 32/32 | 32/32 | 32/32 |
| replace | 1.0 | 32768 | 0 | 32/32 | 32/32 | 32/32 |
| wrong_anchor | 1.0 | 0 | 0 | 0/32 | 32/32 | 0/32 |
| wrong_anchor | 1.0 | 32768 | 0 | 0/32 | 32/32 | 0/32 |

门槛：`{"large_joint_recovery": true, "long_content_preserved": true, "short_joint_preserved": true}`。

recovery：配对净增 32/32；差值比例的 bootstrap 95% CI：[1.0, 1.0]。

replacement_vs_lowrank：配对净增 0/32；差值比例的 bootstrap 95% CI：[0.0, 0.0]。

replacement_vs_reminder：配对净增 0/32；差值比例的 bootstrap 95% CI：[0.0, 0.0]。

french_vs_english_anchor：配对净增 32/32；差值比例的 bootstrap 95% CI：[1.0, 1.0]。

## delay

结果行数：192；完整：True；模型复核完整：True；判定：`stop_current_configuration`。

| 方法 | 强度 | 参考长度 | 延迟 | 法语遵循 | 内容正确 | 联合成功 |
|---|---:|---:|---:|---:|---:|---:|
| add_lowrank | 1.0 | 32768 | 512 | 32/32 | 32/32 | 32/32 |
| add_lowrank | 1.0 | 32768 | 4096 | 32/32 | 32/32 | 32/32 |
| baseline | 0.0 | 32768 | 512 | 0/32 | 32/32 | 0/32 |
| baseline | 0.0 | 32768 | 4096 | 0/32 | 32/32 | 0/32 |
| replace | 1.0 | 32768 | 512 | 31/32 | 32/32 | 31/32 |
| replace | 1.0 | 32768 | 4096 | 11/32 | 32/32 | 11/32 |

门槛：`{"immediate_gain_positive": true, "retains_half_gain": false, "long_content_preserved": true}`。

延迟 512 token：投影恢复相对同延迟基线的联合成功净增 31/32；95% CI：[0.90625, 1.0]。

延迟 4096 token：投影恢复相对同延迟基线的联合成功净增 11/32；95% CI：[0.1875, 0.5]。

## 实测计算耗时

下面是各阶段较慢 worker 的实测运行时间（含加载、缓存检查、预填与分支生成），不含阶段间的模型审阅和人工审核。

| 阶段 | 较慢 worker（秒） | 两个 worker 合计（秒） |
|---|---:|---:|
| baseline | 146.3 | 291.7 |
| tune | 254.9 | 506.8 |
| compare | 314.6 | 620.0 |
| delay | 254.1 | 498.3 |

## 审计与限制

- 原始回答、原始空白盲审导出、单独的模型复核标签均保留；每条模型标签明确记录 `human_reviewed=false`。
- 审阅导出隐藏方法和参数，但协调者了解实验且见过基线汇总，因此不是独立人工盲审。正式结论待人工复核。
- 全部阶段保留同一冻结数据、源码、模型配置和 runtime 指纹；没有按测试集结果重新调参。
- 服务器 GPU 与其他任务共享，下面的耗时是该运行环境的观察值，不作为专用 GPU 的性能基准。
- BF16 缓存续接数值阈值曾做本机调整，并保留 KL、首 token、独立缓存分叉与干预有效性检查。详见 `operations/20261016/bf16-adjustment.md`。
- 本先导结果仅适用于当前模型、语言策略、资料与任务条件；不证明行为子空间专属性或机制。

数据目录：`/mnt/mydata/wja/rcuf/data/pilot-20261016`。

完整结果与置信区间：`/mnt/mydata/wja/rcuf/runs/pilot-20261016b-2gpu/pilot-result-model-reviewed.json`。

审阅来源：`/mnt/mydata/wja/rcuf/operations/20261016/continuation`。
