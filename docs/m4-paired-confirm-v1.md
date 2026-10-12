# M4：M0 与 M3 的全新题配对复验

## 预先固定的协议

1. **问题。** M3 在原固定验证题上从 2/200 增至 12/200；在未用于此前训练、选题或验证的新题上，严格正确数的方向是否重复？
2. **模型。** 比较 M0 两次更新后的检查点与 M3 十次更新后的检查点，分别核验公开 SHA-256 清单。使用同一个 pinned 原始 tokenizer、M3/M0 相同的系统提示词、128-token 上限、贪心解码和严格 verifier。
3. **题目。** 仅从 pinned GSM8K `main/train` 的原训练分区选择。排除 M0 原 200 题验证分区、此前诊断/扫描题、M1 的开发/保留题、M2 两组题、M3 全部 640 道训练候选与 200 道验证题。对剩余 ID 按 `SHA-256("m4-paired-confirm-v1:" + prompt_id)` 升序取 **400 道**，在生成任何回答前保存题目清单。
4. **主指标。** 两份权重在同一 400 题上的严格正确数，并列出“两者都对、仅 M3 对、仅 M0 对、两者都错”。同时报告 `missing_final_answer`、`incorrect_answer`、`invalid_final_answer` 和截断数。
5. **次要归因。** 所有回答可计算末尾数字的机械 proxy，但它不等于可靠的答案值；人工复核严格正确性发生翻转的题，区分纯格式修复、数值改变和未完成计算。若无法可靠识别候选数值，明确弃判。
6. **结论边界。** 这是 train 来源的独立保留样本，不再运行官方 test，也不更新权重。若严格增益重复，仍需按格式与数值分别表述；若没有重复，不把原 2/200 → 12/200 推广到新题。

## 结果与结论

预结果协议和 400 道题清单先以提交 `ca8d104` 同步至公开仓库，随后才生成 M0/M3 回答。两份本地检查点均通过公开 SHA-256 清单核验。MPS float32、相同原始 tokenizer、相同提示词、128-token 贪心解码和严格 verifier 完成全部 800 次推理；无权重更新。

| 同一 400 道题 | M0 权重 | M3 权重 |
|---|---:|---:|
| 严格正确 | **8/400（2.0%）** | **30/400（7.5%）** |
| 无合法最终答案行 | 356 | 270 |
| 有合法最终行但数值错误 | 34 | 96 |
| 最终行格式无效 | 2 | 4 |
| 长度截断 | 124 | 121 |

配对四格表：两者都对 **7**；仅 M3 对 **23**；仅 M0 对 **1**；两者都错 **369**。净增 **22/400，即 5.5 个百分点**。M3 原验证题上的增益为 2/200 → 12/200；在本次不重叠的题上，严格分数提升方向再次出现。这支持**当前两份权重在两批固定保留题上的严格 verifier 表现差异**，不等于数学推理普遍提升。

对 24 道严格正确性翻转题的人工归因：23 道新增正确中，**15** 道 M0 原回答已有正确数值、但缺少有效最后一行；**7** 道 M0 原数值错误；**1** 道原回答达到长度上限、没有可靠最终数值。唯一变错题是 M3 将原先正确的 `770` 写成错误的 `475`。例如 `train/04989` 的 M0 回答已有 `#### 15`，随后又输出模板文字，导致最后一行不合法；M3 补出有效的末行。`train/04661` 的 M0 则停在错误的 `$45`，M3 继续计算为 `$77`。`train/01922` 的 M0 回答被截断，因此不能把片段里的中间数字当作最终答案。

还有 **68** 道题从 `missing_final_answer` 变为 `incorrect_answer`：格式变得可判定，也会暴露错误数值。对全部回答计算的机械“末尾数字与标答相同”代理值，加上严格正确，M0 为 **60/400**、M3 为 **62/400**；末尾数字可能是中间量或模板残留，**不能把 60→62 当作答案值准确率**。本次人工归因只覆盖 24 道严格正确性翻转题，不能据此推断其余题的推理正确率。

## 运行与证据

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.evaluate_m4_paired prepare \
  --output-dir results/m4/my-paired-confirm
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.evaluate_m4_paired run \
  --output-dir results/m4/my-paired-confirm
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.audit_m4_flips \
  --run-dir results/m4/paired-confirm-v1 \
  --output-dir results/m4/my-flip-audit
```

需保留 M0/M3 本地检查点、pinned 模型与 GSM8K 缓存，推理需要 MPS。`prepare` 在生成前保存[题目清单](../results/m4/paired-confirm-v1/manifest.json)并核对来源；`run` 在已冻结清单上逐题写入 [M0](../results/m4/paired-confirm-v1/m0.jsonl)和 [M3](../results/m4/paired-confirm-v1/m3.jsonl)原文，并在两臂完整且重判一致后写[配对汇总](../results/m4/paired-confirm-v1/summary.json)与[逐题配对记录](../results/m4/paired-confirm-v1/paired.jsonl)。[翻转题标注](../results/m4/paired-confirm-v1/flip-audit/annotations.jsonl)及其[校验汇总](../results/m4/paired-confirm-v1/flip-audit/output/summary.json)可以独立审阅。生成中断后只从已核对的完整前缀续跑；已有完成汇总的目录不可再次运行。

## 实现理解交接

1. **Outcome.** 在全新 400 道题上复现严格正确数的正向差异：M0 8、M3 30，配对净增 22。
2. **Before.** M3 的 200 道固定验证题上是 2→12；其中相当部分是格式修复。M4 前没有在另一批新题上配对复验。
3. **Why（Supplied）.** 用户同意在同一批新题上比较 M0 与 M3，并单独观察格式和数值变化。
4. **Core intuition.** 每道题同时给两份权重；每题的两种结果构成一对，23 道变对和 1 道变错决定净增 22。
5. **Flow.** [协议配置](../configs/m4-paired-confirm-v1.yaml)固定盐与 400 题预算；[评估入口](../scripts/evaluate_m4_paired.py)核验血缘、旧题排除和检查点，再按 manifest 顺序对 M0/M3 调用原 `evaluate_model()`；[翻转审计](../scripts/audit_m4_flips.py)复核保存回答、标注和原 verifier。
6. **Explained change.** 新入口把既有数据加载、贪心评估和严格判分连接成两阶段流程：先冻结题目并公开，再读取两份权重逐题保存，最后只有在两臂 ID 与判分完全对齐时汇总。训练入口、参数与原 verifier 没有改变。
7. **Invariants and boundaries.** 400 个新 ID 与 1,494 个旧实验相关 train ID 隔离；两臂提示词、tokenizer、生成上限及 verifier 相同。数据源仍是 GSM8K train，官方 test 未在本次运行中使用。
8. **Failure and recovery.** 配置/权重哈希、数据来源、题目交叉、保存行的 ID 或判分不符即停止。中断后可从严格核验过的 JSONL 前缀继续；没有完整 `summary.json` 的运行不能报告为完成。
9. **Evidence（Independently verified）.** 两个检查点哈希通过；400 个选题 ID 唯一且与 1,494 个排除 ID 无交集；800 条输出重新按 pinned 标答与 verifier 核对；24 道翻转标注覆盖完整且引用原文。原始 JSONL、配对汇总、审计输出和测试结果在上述链接与仓库测试中可核查。机械末尾数字仅是 proxy。
10. **Comprehension check.** 见下方五个问题及答案。

### Questions

1. 为什么只比较两次总分，不能替代逐题配对？
2. 400 道题如何避免与 M3 训练和验证题重叠？
3. 为什么 M0/M3 的 `run_official_test_once` 标记不同，仍可使用相同评估生成设置？
4. 运行在第 300 道题中断，什么条件下能续跑？
5. 8/400 → 30/400 与 60/400 → 62/400 分别能说明什么？

### Answers

1. 总分不能显示同一题的变对和变错；本次是 23 对 1，净增 22。
2. 选题前取旧扫描/诊断、M1、M2、M3 的全部候选和验证 ID 的并集，从 pinned train 原训练分区剩余题中按固定哈希盐选 400。
3. 它是运行官方 test 的流程标记；本次实际生成只用两者一致的提示词、`do_sample=false` 和 `max_new_tokens=128`，且 M4 自身不加载官方 test。
4. 先核对冻结 manifest、检查点、已保存行的题目顺序和严格判分，再从完整前缀之后继续；损坏行不可被当成完成。
5. 前者证明本次固定题上的严格判分提升；后者只是机械末尾数字代理值，不能证明数学答案值或推理能力提升。
