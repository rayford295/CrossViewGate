# CrossViewGuard 修订：从标签拟合到视角条件可证实性

- **Date:** 2026-07-10
- **Status:** **current-data execution endpoint reached; the revised central
  hypothesis is incomplete and unconfirmed.** RQ1 Study A met 0/3 development
  criteria. Study B is executed to the Eaton data boundary but inferentially
  blocked. RQ2/RQ3 are strict dependency NO-GO because real field-level damage,
  view-conditional observability/judgment/abstention, and held-out attestation
  labels do not exist. The four-role CVIAN label-cost selector is a
  hash-chain-verified consumed development/exploratory within-CVIAN NO-GO. The
  one-time Milton run is sensitivity-only: relative geometry is descriptively
  positive against random, while intervals against farthest and clockwise cross
  zero. See the consolidated
  [`execution status`](results/crossviewguard_execution_status_20260710.md),
  [`CVIAN anatomy`](results/disagreement_anatomy_cvian_v1.md),
  [`Eaton feasibility audit`](results/disagreement_anatomy_eaton_v1.md), and
  [`attestation dependency gate`](results/attestation_dependency_gate_20260710.md).
- **Relationship to prior plan:** 本文修订
  [`2026-07-10_crossviewguard_active_evidence_research_plan.md`](./2026-07-10_crossviewguard_active_evidence_research_plan.md)。
  原计划的全部 implementation update、P0 修复、风险协议与 sequential-reveal
  基础设施继续有效；修订只针对研究对象、效用函数与叙事重心。

## 修订动因

两个独立信号指向同一个结论：

1. **导师判断：** 不同视角对同一建筑给出不同 severity 分类可能不是误差——
   街景看立面、门窗、外墙，遥感看屋顶与结构轮廓。一个一维 severity 标签是对
   多面证据的**有损投影**；需要关注的信息超过标签本身。
2. **Phase-1 NO-GO 的另一种解释：** learned selector 以 label cost 为 utility
   target（`loss_before - loss_after - acquisition_cost`），未超过几何基线。
   除了"selector 不成熟 + 协议债"之外，存在一个此前未检验的对立假说——
   **目标函数错位**：如果标签是视角依赖的投影，label cost 本身就是带结构噪声
   的监督信号。

## 修订后的中心命题

> Cross-view disagreement encodes what each view can attest. We hypothesize it
> is systematically produced by view-conditional attestability, and that
> acquiring evidence to complete attestation coverage — not to force label
> agreement — yields a risk-controllable assessment workflow.

原命题（"分歧应触发行动"）保留为该命题的推论。

## 研究问题

**RQ1 — 分歧的成因结构（解剖学）。** 跨视图 severity 分歧能否被"损伤所在
部件的视角可见性"系统性解释？
- Study A（CVIAN，可立即执行）：分歧方向与街景可见度特征的关联结构；
  见 [`docs/results/disagreement_anatomy_protocol.md`](results/disagreement_anatomy_protocol.md)。
- Study B（Eaton，可实现性审计已执行、主检验受阻）：屋顶主导损伤是否产生"遥感报重、街景报轻"
  的定向分歧，立面主导损伤反之；可被部件可见性解释的分歧份额。

**RQ2 — 视角条件可证实性。** 每个视角对每个损伤要素（roof / eaves / siding /
window / deck / severity）的可证实能力能否量化为可靠的 view × field 证实
矩阵并支持带 abstain 的字段级判断？该矩阵是否由空间几何（方位、距离、遮挡、
建筑朝向）可预测？

**RQ3 — 证据覆盖驱动的主动获取。** 以"证实尚未被证实的决策相关字段"为效用
的 next-view 策略，在相同视角预算下能否超越几何基线，并优于以标签损失为目标
的 phase-1 selector？这是对 NO-GO 两种解释之间的直接反事实检验。

**RQ4 — 风险可控的停止与升级。** 基于证实状态的 stop / defer_human 规则能否
在预注册风险上界下降低 severe miss 与人工成本的复合损失？跨事件迁移时保证
如何退化？

RQ 之间严格递进：RQ1 不成立则方向证伪（先做、便宜）；RQ1 成立才投入 RQ2 的
矩阵；RQ3 用矩阵反攻 NO-GO；RQ4 落到风险可控工作流。每一步有明确证伪出口。

## 对原计划的五处具体修订

1. **中心命题**：如上改写；"分歧应触发行动"降为推论。
2. **WS2 效用函数**：`label-loss reduction` 替换为 `attestation-coverage gain`
   （按成本本体加权的"新证实的决策相关字段数"）。原 label-cost selector 保留
   为必须对比的基线。Go/No-Go 判据需围绕覆盖效用重新预注册。
3. **WS5 提前并更名**：`Task-Conditioned Evidence and Grounded Reports` →
   核心工作流 `View-Conditional Attestability Matrix`。VLM 自由文本继续排除
   （原第 9 节禁令不变）。
4. **新增 RQ1 解剖学里程碑**：作为下一阶段第一个执行项；Study A 用既有
   per-sample evidence 表（p0.2-v1）即可运行。
5. **风险认证主场 Ian → Eaton**：修复后 Ian 只有 4 个 sequence-connected
   依赖组、severe 校准样本仅落在 2 个 block，五种子全部只能 `risk-aware`。
   Eaton 有 19,776 条已连接记录与充足 block，conformal 上界才可能可认证。
   Ian 保留为 in-event 迁移压力测试。

## P0.4 补充条款（必须先于 RQ2 执行）

原 P0.4 要求"将 inspector-recorded attributes 与 visually observable evidence
分开"。Eaton 可实现性审计现已证明，本地 DINS component-like fields 是
construction/material/presence/exposure，而不是 component-damage reference；
它们不能直接作为可证实性真值。只有未来取得 provenance-bearing damage
annotations 后，以下规则才适用：

- 检查员字段只有在 provenance 明确证明其语义为"该部件损伤是否存在/主导"时
  才能作为参照；
- 必须先建立 **image-visible field whitelist**：逐字段判断其在街景/遥感分辨率
  下是否原理上可见（例如 vent screen 细节在街景采集距离外不可见，应排除或
  仅作 stress test）；
- 白名单的制定与冻结先于任何 RQ2 矩阵训练，避免把"图像看不见"与"模型没
  学会"混淆。

## 执行结果与下一合法阶段

原“下一阶段”清单中，凡当前资产可支持的项目已经执行到证据边界：

1. **CVIAN label-cost active-view：NO-GO。** 四角色实验未超过 farthest 或
   privileged maximum-building；Windows CRLF 缺陷经独立复核后不改变结果，但
   claim 降级为 hash-chain-verified consumed development/exploratory
   within-CVIAN NO-GO。
2. **Milton transfer：sensitivity complete, no confirmation。** 冻结的一次性
   zero-shot scoring 已完成；relative 对 random 的 dependency-group 与
   spatial-block 描述性差值为 `+0.024557`（95% CI
   `[0.013279, 0.044898]`）和 `+0.021502`（95% CI
   `[0.013730, 0.034251]`），但对 farthest/clockwise 的区间跨零，不能产生 GO。
   P0.3 的 collected-vs-generated post-SVI 媒体来源仍未解决；manifest/hash
   verification 只验证本地评分链，不能补足该 provenance。
3. **RQ1 Study A：0/3。** 定向不对称、visibility-conditioned correctness、
   held-out correctness prediction 三项预设标准均未满足；5/5 seed 的弱方向趋势
   未确认，且 7 个 spatial blocks 不能支持 in-event confirmation。
4. **RQ1 Study B：BLOCKED。** Eaton 已连接 19,776 个 DINS rows，但现有字段是
   construction/material/presence/exposure，而非 component-damage dominance；
   真实 per-view severity predictions 与冻结空间组也不存在。这是构念/依赖阻塞，
   不是负经验结果。
5. **RQ2/RQ3：strict dependency NO-GO。** 不创建伪标签、不训练无效矩阵，也
   不把现有 label-cost utility 改名为 attestation coverage。RQ4 同时因缺少
   上游 attestation state 和独立风险校准而受阻；现有 stop/defer 仅为
   risk-aware heuristic。

下一合法阶段不是继续消耗 CVIAN/Milton v1，而是取得新证据并重新冻结协议：

- Eaton 需要 provenance-bearing `component_dominance ∈
  {roof,facade,mixed,unknown}`（语义必须是 damage dominance）、唯一
  `(seed,pair_id)` 双视角预测和冻结 spatial block；
- RQ2 需要满足
  [`attestation contract`](../configs/crossviewguard_attestation_contract_v1.json)
  的 field × view × role 标注、支持度/功效审计，以及距离、方位、朝向、遮挡等
  几何信息；
- RQ3 需要 out-of-role calibrated RQ2 matrix、测试前冻结的字段权重/成本、互斥
  policy-fit/validation/test dependencies，以及 prospective 或外部
  attestation-labeled test；
- RQ4 需要足量独立 calibration groups/severe cases，并预注册 loss、`alpha`、
  `delta`、threshold grid 与跨事件校准/shift protocol。

## 当前可发表分支与边界

当前数据分支已落入预先声明的 fallback，而不是完成了中心命题。可发表产出为：

- 严格协议的 sequential-reveal benchmark（隐藏视角隔离、fail-closed 校验、
  消耗即作废纪律）；
- oracle gap 的存在性证明与可实现上限的诊断；
- RQ1 的分歧解剖学结论（无论方向）；
- 修复后的 CVIAN 空间划分及其对 legacy split 虚高的量化；
- risk-aware selective triage 与 routing 产物。

即 benchmark + protocol + negative/dependency anatomy 论文。它可以诚实报告
当前 selector NO-GO、Milton sensitivity、RQ1 0/3 和 RQ2/RQ3 数据门禁，但不能
包装成“attestability 已验证”或“attestation coverage 已优于几何基线”。只有
新标注/新事件满足上述 unlock 条件后，中心假说论文分支才可重新打开。

## Claim 纪律（沿用并扩展）

- 原计划的 consumed-test 规则、三角色分离、`risk-controlled`/`risk-aware`
  降级规则全部沿用；
- RQ1 Study A 的全部当前结果标注 **exploratory / development**：分析只使用
  `gate_fit` 与 `risk_calibration` 角色行，**不触碰 `final_test`**；
  由于当前仅 7 个 spatial blocks，不能为了追求显著性而消耗 `final_test`；只有
  新的功效充分、依赖隔离协议冻结后才允许一次性确认；
- "视角合理分歧"与标注噪声的区分以 RQ1 的可解释份额为准，禁止在未量化前
  以"分歧即信息"作为结论性表述；
- CVIAN four-role 与 Milton v1 均不得继续用于 selector、预算、阈值或 GO 规则
  选择；
- RQ2/RQ3 的 dependency gate 是 upstream BLOCKED/NO-GO，不是经验上证明中心
  假说为假；中心命题必须写作 **incomplete / unconfirmed**。
