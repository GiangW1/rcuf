# 4B BF16 缓存检查的本机数值校准

原始源码在两张 5090 上运行真实 Qwen3.5-4B 时，缓存独立分叉可复现，分段与整段预填的首 token 一致，但原始 `atol=0.15, rtol=0.02` 检查失败，最初运行最大 logits 差异为 0.21875。原始失败运行保存在 `runs/pilot-20261016-2gpu`，没有正式 baseline 结果。

使用同一短提示，在 GPU 7 上比较 BF16/FP32、FLA/PyTorch 参考实现，诊断记录保存在 `cache-diagnostics/report.json` 和四份原始 logits 文件中：

| 精度 | GDN 后端 | 最大 logits 差异 | RMS 差异 | KL(full‖continued) |
|---|---|---:|---:|---:|
| BF16 | FLA | 0.3125 | 0.03027 | 0.001660 |
| BF16 | PyTorch | 0.3125 | 0.03857 | 0.0001273 |
| FP32 | FLA | 0.008967 | 0.0008565 | 0.0000008530 |
| FP32 | PyTorch | 0.00004101 | 0.000003701 | 0.00000005364 |

四种条件的首 token 都为 43（`L`）。FP32 参考实现结果支持 BF16 舍入与计算分块引入的数值误差解释；这项检查不构成实验收益结论。

本地调整：BF16 的逐 logit 绝对容差由 0.15 改为 0.25，保留相对容差 0.02；额外要求 `KL(full‖continued) <= 0.01`，并继续要求首 token 完全一致。FP32 保留原来的逐 logit 容差。相同缓存分叉 logits 的完全复现、状态改动影响新 logits、FP32 循环状态布局等其余检查保持执行。每个 worker 的 `cache-checks-*.json` 记录实际误差、容差和 KL。

新增四项回归检查，验证低概率 token 的 BF16 舍入可通过、FP32 原阈值仍有效、逐 logit 接近但概率分布明显漂移会失败、贪心首 token 改变会失败。修改后完整 CPU 测试为 39 passed / 3 skipped，并重新运行 GPU 6 和 GPU 7 的实际 CUDA/FLA 检查。

调整发生于正式 baseline 之前，未改变模型权重、输入、数据划分、rank、方法强度或实验门槛。正式运行使用新目录 `runs/pilot-20261016b-2gpu`，源代码指纹与旧尝试分开保存。
