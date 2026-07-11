# CrossViewGuard 修订：从标签拟合到视角条件可证实性

- **Date:** 2026-07-10
- **Status:** adopted direction revision; supersedes the *framing* of the
  active-evidence plan while keeping its infrastructure, protocols, and P0
  repairs intact. RQ1 Study A (CVIAN directional anatomy) is **executed
  (exploratory)**: none of the three development criteria was met — the
  existing four visibility features carry no held-out attestability signal,
  a consistent but unconfirmed street-more-severe directional lean exists in
  5/5 seeds, and the 7-block cluster count makes in-event confirmation
  structurally impossible. Study B (Eaton component anatomy) is now the
  critical test; it awaits local access to the `dins_field_join` derived
  artifacts. See
  [`docs/results/disagreement_anatomy_cvian_v1.md`](results/disagreement_anatomy_cvian_v1.md).
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
- Study B（Eaton，待 DINS 数据）：屋顶主导损伤是否产生"遥感报重、街景报轻"
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
分开"。新框架以 DINS 部件字段为可证实性参照，与该警告不矛盾，但需要显式
规则：

- 检查员字段只在"该部件损伤是否存在"层面作为参照；
- 必须先建立 **image-visible field whitelist**：逐字段判断其在街景/遥感分辨率
  下是否原理上可见（例如 vent screen 细节在街景采集距离外不可见，应排除或
  仅作 stress test）；
- 白名单的制定与冻结先于任何 RQ2 矩阵训练，避免把"图像看不见"与"模型没
  学会"混淆。

## 执行顺序（下一阶段）

按信息价值/成本比排序，前三步都便宜且每一步都可能改变后续步骤的性价比：

1. **Milton post-SVI provenance 审计**（原 P0.3 遗留）——它是最可能的
   confirmatory transfer 路径，若含生成数据需要先降级其用途；
2. **统计功效估算**——在切新 holdout 之前，回答 in-event CVIAN 的依赖组数量
   是否原理上撑得起 bootstrap CI 下界 > 0 的 confirmatory 判据；若否，
   confirmatory 主场直接规划为 cross-event；
3. **RQ1 Study A**（立即可执行）+ **可实现上限诊断**：训练可见全部非隐藏视角
   特征的"作弊 selector"模仿 oracle——若它也逼近不了 0.344，则 phase-1 gap
   的大部分在因果上不可实现，RQ3 的预期收益需要重新定标；
4. **RQ1 Study B**（待 DINS 数据可达）；
5. **RQ2 矩阵 → RQ3 覆盖效用 active-view**（新 sequence/event holdout +
   base-encoder/OOF role isolation，按原计划既定要求）；
6. **RQ4 风险认证**（Eaton 主场）。

## Fallback 论文分支（预先声明）

若 RQ3 在第二次 confirmatory 尝试中仍为 NO-GO，本项目的可发表产出为：

- 严格协议的 sequential-reveal benchmark（隐藏视角隔离、fail-closed 校验、
  消耗即作废纪律）；
- oracle gap 的存在性证明与可实现上限的诊断；
- RQ1 的分歧解剖学结论（无论方向）；
- 修复后的 CVIAN 空间划分及其对 legacy split 虚高的量化；
- risk-aware selective triage 与 routing 产物。

即 benchmark + protocol + anatomy 论文。预先声明此分支的目的，是避免届时被
沉没成本推动继续调参。

## Claim 纪律（沿用并扩展）

- 原计划的 consumed-test 规则、三角色分离、`risk-controlled`/`risk-aware`
  降级规则全部沿用；
- RQ1 Study A 的全部当前结果标注 **exploratory / development**：分析只使用
  `gate_fit` 与 `risk_calibration` 角色行，**不触碰 `final_test`**；
  final_test 上的一次性确认要等分析代码冻结后按协议执行；
- "视角合理分歧"与标注噪声的区分以 RQ1 的可解释份额为准，禁止在未量化前
  以"分歧即信息"作为结论性表述。
