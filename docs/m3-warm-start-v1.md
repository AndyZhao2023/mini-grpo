# M3：从 M0 权重重启优化器的 10 步 GRPO 实验

## 预先固定的实验协议

1. **目标。** 检查在 M0 最终权重上再做 10 次新鲜 rollout 与 GRPO 更新，能否提高同一批新验证题上的严格 `#### <number>` 正确数。
2. **起点。** 加载 M0 `step-000002` 本地权重并校验公开 SHA-256 清单；创建新的 AdamW。M0 没有保存优化器状态，因此本实验不是原运行严格连续的第 3 至 12 步。
3. **题目。** 从固定 GSM8K train 数据中排除原 M0 验证题、M0 早期扫描和 M1/M2 用过的训练题。在其余题中按题干词数和 ID 取最短的 640 道作为训练候选；再从剩余题中按 `m3-warm-start-v1-held-out:prompt_id` 的 SHA-256 顺序固定 200 道验证题。选择和排除的 ID 均写入 `dataset-split.json`。
4. **更新。** 保持 M0 原提示词、`G=4`、256 个生成 token 上限、严格 verifier、`lr=1e-7`、clip epsilon 0.2、`beta=0` 和梯度裁剪 1.0。每次从当前 policy 新采样，扫描最多 64 个此前未扫描过的训练题，找到第一组混合奖励后更新一次。预定上限 10 次更新；找不到混合组或更新后 `approx_kl_old > 0.05` 时停止并标记未完成。
5. **评估。** 更新前后均用相同 200 道新验证题、贪心解码、128-token 上限和严格 verifier。单独列出变对与变错题，不把训练 reward 或混合组比例当成泛化准确率。官方 test 不再使用。

## 运行与证据

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.run_warm_start \
  --config configs/m3-warm-start-v1.yaml \
  --output-dir results/m3/warm-start-v1
```

运行入口在 [`scripts/run_warm_start.py`](../scripts/run_warm_start.py)，底层闭环复用 [`scripts/run_m0.py`](../scripts/run_m0.py)。[配置](../configs/m3-warm-start-v1.yaml)固定起点、预算和数据选择；模型检查点只保留本地，不提交到 Git。

## 结果

1. **Outcome.** 完成 **10/10** 次新鲜 rollout 与 GRPO 更新。新验证集严格正确从 **2/200（1%）升至 12/200（6%）**：10 道变对、0 道变错、2 道始终正确。
2. **Before.** M0 原内部验证为 9/200，M2 在固定 M0 权重上得到最短题优先 11/64 个混合组，但没有训练。M0 权重保留本地，AdamW 状态未保存。
3. **Why（Supplied）.** 用户同意从 M0 权重继续，重新初始化优化器，并用更多更新检验固定验证题表现。
4. **Core intuition.** M0 权重提供模型起点；每次更新必须从当前 policy 新采样一组有正负相对优势的回答。M2 旧采样不能当作更新后的 on-policy 数据。
5. **Flow.** 排除 654 个旧实验涉及的 train ID，预定 640 道最短训练候选和另外 200 道哈希选择的验证题。第 1 步扫描 3 道题，选中 `gsm8k/main/train/06019`，奖励 `[0,0,1,0]`；后续九步都重新采样，然后在原 200 道新验证题上复评。
6. **Explained change.** [新入口](../scripts/run_warm_start.py)核对 M0 来源、权重、数据与旧题，加载模型并创建新 AdamW，再调用原 [`run_cycles()`](../scripts/run_m0.py)。训练循环新增可选的“已扫描题不再扫描”和自定义数据 manifest，M0 原配置仍沿用旧行为。[配置](../configs/m3-warm-start-v1.yaml)、[题目 manifest](../results/m3/warm-start-v1/dataset-split.json)、[逐步记录](../results/m3/warm-start-v1/steps/step-000001/step.json)、[汇总](../results/m3/warm-start-v1/summary.json)和[权重校验清单](../results/m3/warm-start-v1/checkpoint-sha256.txt)构成证据链。
7. **Invariants and boundaries.** Prompt、`G=4`、256-token 训练生成上限、reward、clip epsilon 0.2、`beta=0`、`lr=1e-7` 和梯度裁剪 1.0 沿用 M0；验证仍为 128-token 贪心解码。训练、验证、M0 原验证、M1/M2 旧题互不重叠。官方 test 未运行。本次不是继承旧优化器状态的严格续训。
8. **Failure and recovery.** 来源哈希不符会在训练前失败；找不到混合组或更新后 `approx_kl_old > 0.05` 会标记未完成并停止。中断后没有完整 summary 的结果不得当作 10 步结论。检查点不含优化器状态，需从 M0 权重在新目录重跑；MPS 不承诺字节级相同。
9. **Evidence.** 本机 MPS、float32、PyTorch 2.14.0、Transformers 4.57.6 完成运行，82 项测试通过。共扫描 30 道互不重复的训练题，选中 10 个互不重复的混合奖励组；最大更新后 `approx_kl_old=0.00997`。独立按 pinned 标答重算前后 400 条验证输出与入训的 40 条回答，reward/reason 全部吻合；最终权重的 12 个文件通过 SHA-256 检查。重载最终权重后，前两道验证题的完整输出与保存记录一致；保存的 tokenizer 与原 pinned tokenizer 对全部 200 道验证题产生相同的 prompt token ID。10 道变对题此前均正常结束但缺少合法最终行，因此严格正确数的改善不能单独证明算术推理能力提高。
10. **Comprehension check.** 见下方五个问题与答案。

| 指标 | 更新前 M0 权重 | 更新后 M3 权重 |
|---|---:|---:|
| 新验证题严格正确 | 2/200 | **12/200** |
| `missing_final_answer` | 182 | 137 |
| `incorrect_answer` | 11 | 43 |
| 长度截断 | 63 | 59 |
| 平均生成 response token | 73.52 | 75.105 |

从仓库根目录执行 `shasum -a 256 -c results/m3/warm-start-v1/checkpoint-sha256.txt` 可检验本次保留在本地的最终权重。新运行请使用尚不存在的输出目录；若本地没有 M0 权重，先按 M0 文档重跑生成。

## Questions

1. 为什么这个实验从 M0 权重开始，却不能称为原运行严格连续的第 3 步？
2. 第 2 次更新所用的 rollout 为什么必须重新生成？
3. 640 道训练候选题和 200 道新验证题如何保持隔离？
4. 若第 6 步找不到混合奖励组，应该怎样报告和恢复？
5. `2/200 → 12/200` 是否证明模型的算术推理普遍提升？

## Answers

1. M0 没有保存 AdamW 的动量等状态；M3 加载模型权重，但重新初始化优化器。
2. 第 1 次更新改变了 policy；旧 rollout 的 `old_log_probs` 属于旧 policy，下一次 on-policy 更新需要当前模型的新采样。
3. 先排除旧实验 ID，固定 640 道最短题，再仅从剩余题用固定哈希盐选择 200 道验证题；manifest 保存全部 ID。
4. 标记未完成并保留部分证据；因优化器状态未保存，要从 M0 起点在新目录重跑。
5. 不能。它支持本次固定题集上严格正确数的配对提升；单种子、200 题、明显的输出格式变化和未重复官方 test 限制了更广泛的结论。
