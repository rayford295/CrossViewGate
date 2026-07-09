# From Classification to Actionable Reports: 研究方向深度思考

**Date:** 2026-07-09
**Status:** direction memo — 思考记录，未执行
**Context:** CrossViewGate 目前落脚在灾害分类（reliability-gated cross-view damage assessment）。本文回答：能否用 cross-view 产出不止是分类，而是类似 report 的、更可执行的解决方案？

---

## 一句话结论

**可行，而且是当前最顺的下一步 —— 但"用 VLM 生成灾害报告"本身在 2025–2026 已经不新了。差异化不在"生成报告"，而在 CrossViewGate 独有的 reliability/conflict 信号：做"可信度条件化的可执行报告"（reliability-conditioned actionable reporting），报告里每条论断带证据来源、可信度和下一步行动。**

---

## 1. 资产盘点：gate 输出天然就是"报告原料"

| Gate 已有的输出 | 报告中的角色 |
|---|---|
| Per-sample gate 权重（信街景还是信卫星） | "证据来源"字段 —— *为什么*下这个结论 |
| 11 个可解释特征（建筑可见度、熵、视角分歧） | 可读的 evidence 描述（"街景中建筑居中且模型置信"） |
| Conflict density 地图（Spearman r=0.615，免标注） | Tile 级**优先级地图** —— 直接可执行 |
| 校准后的 per-view 置信度 | 报告的置信度字段 |
| Eaton 标签体系（No Damage / Affected 1–9% / Minor / Major / Destroyed / Inaccessible） | 即 **CAL FIRE DINS** 字段体系 —— DINS 结构化巡检记录可直接当"报告 ground truth"，无需自标 |

**最被低估的资产是当前被排除的 `Inaccessible` 类。** 在分类框架里它是废料（类太稀疏）；在决策框架里它是黄金 —— "无法从现有视角判断"本身就是可执行的行动项：**列入人工核查清单，优先派员**。Oracle gap 的发现（冲突样本上单视角仲裁值 0.37–0.41 准确率）讲的是同一件事：**分歧不是噪声，是信息**。

## 2. 竞争格局（2025–2026 必读四篇）

- **DisasterM3**（arXiv 2505.21089）：26,988 组双时相卫星图 + 123k 指令对，任务含 report generation。但**纯 overhead、无 cross-view、无可信度机制**。
- **RAPID**（arXiv 2606.21819，最近的竞品）：多智能体管线，卫星+街景输入，输出损害等级+类型+"location-specific 决策报告"。但是 zero-shot VLM 拼装，cross-view severity 只有 **0.627 准确率**，没有 reliability gating、没有冲突感知、没有 decision-utility 评估。
- **GeoDisaster**（arXiv 2606.17246）：编排式 agent 基准，含诊断报告生成、flood-safe routing —— 社区正把"评估 → 行动"当下一个前沿。
- **Recov-Vision**（arXiv 2509.20628）/ **DisasterInsight**（arXiv 2601.18493）：SVI+VLM 灾后恢复与 function-aware 评估。

**含义：** 直接"接个 GPT 生成报告"会被审稿人打成 engineering wrapper。但这些工作有同一个未解软肋 —— **报告的幻觉与证据可信度无人处理** —— 这恰好是 gate 的主场。

## 3. 三档方案（从小到大，可叠加）

### 方案 A —— 不碰 LLM 的最小闭环："From gating to triage"（约 3–4 个月）

把 gate 输出直接转成决策产品：per-building triage 分数 + tile 优先级地图 + 巡检路径模拟。

- **评估：反事实模拟。** 同样巡检人力预算下，按 conflict-density 排序 vs 随机 / 单视角不确定度排序，比较找到 destroyed 建筑的 recall@k。
- 数据已齐（Eaton/Ian/Milton 三灾种），DINS 记录即模拟地面真值；`analyze_conflict_density_maps.py` 的输出即输入。
- 产出：现有 ISPRS 稿 "operational payoff" 的强化，或投 *IJDRR* 独立短文。

### 方案 B —— 主推：Reliability-gated report generation（6–9 个月）

**核心科学问题：可信度信号能否可度量地降低 VLM 灾害报告的幻觉？**

1. 定义结构化报告 schema：damage state、per-view evidence、confidence、recommended action、verification-needed 标志 —— 每个字段可验证，不是自由文本；
2. VLM 读配对图像 + gate 特征生成报告，三组消融：**no-gate / gate-in-prompt / gate-as-router**（低可见度时 VLM 只被允许引用卫星证据，冲突样本强制输出 verification-needed）；
3. 评估：字段级事实准确率（对齐 DINS 字段）+ 幻觉率（报告中无法被任一视角图像支持的论断比例，人工+LLM judge 双评）+ 优先级排序 NDCG + 应急管理专家盲评（TAMU Hazard Reduction & Recovery Center 有真用户）。

**顺手产出：第一个带结构化真值的 cross-view 灾害报告 benchmark**（DisasterM3 是 satellite-only；无人拥有 paired ground+overhead+DINS 字段的报告数据集）。可投 ISPRS JPRS，或做成 benchmark 投 NeurIPS D&B / CVPR EarthVision。

立项句子草稿：

> "Existing VLM disaster reporting systems (RAPID, DisasterM3) generate fluent but unverifiable narratives. We show that visibility-conditioned reliability signals — learned from cross-view conflict — reduce factual errors in generated damage reports by X% and convert model disagreement into actionable verification queues, evaluated against structured DINS inspection records."

### 方案 C —— 长线：Active cross-view agent（proposal 级）

报告不是终点而是决策循环的一环：agent 判断"现有证据够不够下结论"，不够就主动决定下一步 —— 再取一张不同角度街景、请求 UAV、还是转人工。FOV 因果干预结果（building-centered crop 使冲突收益翻倍 0.177→0.365）正是 agent 行动策略的依据。对标 GeoDisaster，适合 NSF proposal 或 dissertation 框架，暂不动手。

## 4. 三个必须提前想清楚的坑

1. **评估别用 ROUGE/BLEU** —— 报告类论文最常见死法。结构化 schema + 字段级验证 + decision utility 才是硬通货。
2. **警惕 ground-truth 循环** —— DamageArbiter 曾踩过 ground-truth leakage 的坑：报告 GT（DINS 字段）与分类 GT 同源，评估设计要确保 VLM 没有间接见过标签。
3. **别稀释现有 ISPRS 稿** —— gate 论文叙事很干净（oracle gap → 线性 gate → 因果机制 → 免标注地图），报告生成应是 follow-up paper。

## 5. 建议排序

**先做方案 A**（成本极低，且反哺 ISPRS 稿一个审稿人无法拒绝的 operational payoff 论据），**同期起草方案 B 的 schema 和 benchmark 设计**。方案 C 留作 proposal。

## References

- DisasterM3: <https://arxiv.org/abs/2505.21089>
- RAPID: <https://arxiv.org/abs/2606.21819>
- GeoDisaster: <https://arxiv.org/abs/2606.17246>
- DisasterInsight: <https://arxiv.org/abs/2601.18493>
- Recov-Vision: <https://arxiv.org/abs/2509.20628>
- Automated Wildfire Damage Assessment via VLMs: <https://arxiv.org/abs/2509.01895>
