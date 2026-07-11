# CrossViewGuard: 从灾害分类到风险可控的主动证据获取

> **Focused Eaton update (2026-07-11):** Eaton now has a frozen four-role
> spatial protocol and fresh role-isolated model execution. The earlier plan
> text saying that Eaton predictions and spatial roles are absent is superseded.
> The focused damaged-component observability v1 branch reaches an
> outcome-blind structural STOP before human annotation because its complete
> development role has 28 spatial blocks, below the registered 30 eligible-block
> minimum; its completed ensemble has 97 unsigned disagreements across 24
> blocks, also below the 250/30 capacity gates. See
> [`results/eaton_component_observability_v1.md`](results/eaton_component_observability_v1.md).

- **Date:** 2026-07-10
- **Status:** **current-data execution endpoint reached (2026-07-11); central
  CrossViewGuard hypothesis remains incomplete and unconfirmed.** The four-role
  CVIAN label-cost selector is a consumed development/exploratory within-CVIAN
  **NO-GO**. The frozen one-time Milton run is sensitivity-only: the relative
  policy has a positive descriptive contrast against random, but its intervals
  against farthest and clockwise cross zero, so it is neither a GO nor external
  confirmation. RQ1 Study A met 0/3 criteria, Eaton Study B is inferentially
  blocked, and RQ2/RQ3 cannot be executed without new provenance-bearing
  annotations. See the consolidated
  [`execution status`](results/crossviewguard_execution_status_20260710.md).
- **Relationship to prior memo:** 本文细化并调整
  [`2026-07-09_from_classification_to_actionable_reports.md`](./2026-07-09_from_classification_to_actionable_reports.md)。
旧 memo 将报告生成视为主要延伸；本文把报告降为呈现层，把
**selective triage、next-best-view 和有限预算派检**定义为核心研究任务。

## 一句话判断

CrossViewGate 的下一步不应是把三分类机械扩展成六分类，也不应止于接入
VLM 生成报告。更强的研究问题是：

> 当街景与遥感证据冲突时，现有证据是否足够，应该信任哪个视角，下一张图
> 应从哪里获取，何时转人工，以及有限巡检资源应该先去哪里？

这一转变把 CrossViewGate 从静态分类器升级为
**风险可控的主动证据调度器**。跨视图分歧不再只产生另一个类别，而是触发
一个可评价的下一步行动。

## 0. 执行终局更新（2026-07-11）

原计划中**依赖当前 CVIAN、Milton 与 Eaton 资产即可合法执行的分支已经跑到
终点**；这不等于 CrossViewGuard 整体完成，也不等于中心命题被证伪。

- CVIAN 四角色 label-cost 实验为 NO-GO：在固定 `k=3` 下，farthest-minus-
  utility 为 `-0.050935`（95% CI `[-0.114026, -0.000229]`），privileged
  maximum-building-minus-utility 为 `-0.025325`（95% CI
  `[-0.054784, 0.001634]`）。
- Windows CRLF 使完成验证器预期的 LF hash 与原始 commitment hash 不同；
  独立复核确认原始 hash DAG、语义指纹和两组 10,000 次 bootstrap 一致。正确
  结论是 **hash-chain-verified consumed development/exploratory within-CVIAN
  NO-GO**，不是 confirmatory result。
- Milton 的一次性 frozen zero-shot sensitivity 已完成。relative 对 random 的
  dependency-group / spatial-block 描述性差值分别为 `+0.024557`
  （95% CI `[0.013279, 0.044898]`）和 `+0.021502`
  （95% CI `[0.013730, 0.034251]`）；对 farthest 与 clockwise 的区间均跨零。
- RQ1 Study A 为 0/3；Eaton Study B 因缺少 component-damage dominance 参照、
  真实双视角预测和冻结空间分组而受阻；RQ2/RQ3 因缺少字段级损伤与可证实性
  标注而为严格 dependency NO-GO。

因此，当前证据只支持“协议、基准、负结果与依赖门禁已经完成”。它**不支持**
“跨视图分歧编码了视角条件可证实性”或“证实覆盖策略优于几何基线”的结论。
重新开启中心假说检验必须先取得新的外部事件/多方位序列和符合
[`attestation contract`](../configs/crossviewguard_attestation_contract_v1.json)
的字段级标注，再冻结新的、角色隔离的 protocol；不得继续在已消耗的 CVIAN
或 Milton v1 test 上调 selector、预算或 GO 规则。

## 1. 为什么这个方向来自现有结果

当前仓库已经给出四项直接支持主动证据获取的结果。

1. **Oracle gap 足够大。** 在 conflict cases 上，正确选择现有单视角模型即可
   额外获得 0.37–0.41 的绝对 accuracy。主要缺口不是更复杂的对称 fusion，而是
   per-sample evidence selection。
2. **可靠性信号已可计算。** 建筑可见度、校准置信度、熵、JS divergence 和
   hard disagreement 已构成 11 维、可解释的 gate 特征。
3. **视角可被主动干预。** Building-centered 90-degree crop 在 Ian 和 Milton
   上均比同几何 random crop 提供更高 fusion gain。目标可见性不仅与结果相关，
   还可以通过改变观察方位来改善。
4. **分歧具有空间效用。** Eaton 的 conflict density 与 tile-level damage
   显著相关，而单视角 uncertainty control 不具备同样信号。这为区域优先级和
   巡检路线提供了起点。

因此，最自然的新任务不是“输出更长的描述”，而是“在风险与成本约束下选择
下一步证据”。

## 2. 当前数据资产与未使用信息

### 2.1 Eaton / Altadena

- 18,428 个 DINS inspection points 和 19,780 个 attachment records；
- 19,746 个完整的 post-disaster street/overhead pairs；
- 13,397 个有效 triplets：pre-disaster overhead、post-disaster overhead、
  post-disaster street view；
- 已有 object id、attachment id、经纬度、remote tile、remote pixel location、
  coverage distance 和 pairing status；
- 本地审计已将 19,776 个 manifest rows 连接到 DINS，覆盖 18,411 个唯一结构。

连接缺口已关闭，但构念缺口没有关闭：当前 roof、eaves、vent screen、exterior
siding、window pane、deck/porch 等字段描述 construction/material/presence/
exposure，不是 component-damage dominance，不能作为 RQ1 Study B 或 RQ2 的
字段级损伤真值。Eaton 还缺真实双视角 severity predictions 与冻结 spatial
groups；详见
[`Eaton feasibility audit`](results/disagreement_anatomy_eaton_v1.md)。

### 2.2 Hurricane Ian / CVIAN

- 4,121 个 post-disaster street/satellite pairs；
- 官方 `02_Position/CVIAN_position.geojson` 已校验 checksum 并与全部本地 pairs
  一对一连接；坐标、空间 block、sequence group 与 boundary-buffer provenance
  已进入修复后的 manifest 和审计协议；
- 五种子 legacy-vs-spatial split 比较已完成，修复后的 macro-F1 按模式下降
  0.049–0.097，确认旧的顺序 `objectid` 分组不能代表干净的空间泛化。

因此 Ian 的地理位置缺口已经关闭；后续受限点不再是坐标，而是独立依赖组数量、
已消耗测试集，以及 RQ2/RQ3 所需的字段级损伤/可证实性标注。

### 2.3 Hurricane Milton

- 源数据包含 pre-disaster street、post-disaster street、post-disaster
  satellite、经纬度、prompt 和 source split；
- 冻结 sensitivity manifest 已保留来源与坐标字段；最终评分包含 1,707 个样本、
  259 个 dependency groups、57 个 spatial blocks、5 个 seeds 与 8 个 origins；
- 当前 `CrossViewTriageDataset` 只读取 post street、post satellite、label 和
  sample id，未使用 pre-disaster street；
- Milton 影响过假说开发，且没有可观测真实 acquisition sequence 或可信 compass；
  因而冻结协议预先将其限定为 sensitivity-only，而不是 confirmatory event；
- **P0.3 媒体来源仍未解决：** manifest/source-split 与 hash-chain 审计证明了本次
  本地评分链的可追溯性，但不能证明 post-SVI 是现场采集还是生成媒体；保留的
  `prompt` 等字段不足以消除 collected-vs-generated ambiguity。

Milton 的一次性 zero-shot sensitivity 与独立完成验证均已结束。它提供 relative
对 random 的正向描述性信号，但没有超过 farthest/clockwise 的确认性证据；不得
重跑以选择 policy 或把它升级为外部验证。完整性边界见
[`Milton integrity verification`](results/active_view_milton_zero_shot_v1_integrity.md)。

### 2.4 已有代码脚手架

仓库已有但尚未形成完整训练流程的组件包括：

- `CrossViewRetrievalDataset`；
- `CrossViewGenerationDataset`；
- `CrossViewTileDataset`；
- Recall@K、地理距离误差和 tile-constrained ranking metrics。

这些组件可用于 pairing QA、cross-view retrieval 和空间候选约束，不必从零
开始设计数据接口。

## 3. 核心任务：Risk-Controlled Active Evidence Acquisition

### 3.1 输入状态

每个样本的状态由以下信息构成：

- 当前可用视图及其 calibrated class probabilities；
- per-view confidence 和 entropy；
- hard disagreement 与 JS divergence；
- street-view building visibility 和 centering；
- 当前已经消耗的视图、人工和移动成本；
- 可用时的 pre-disaster reference、位置和 pairing-quality metadata。

### 3.2 Decision Source and Disposition

系统不再只输出 severity class。为避免把证据来源与工作流动作混为一谈，
每个样本分别输出两个维度。

`decision_source` 表示当前判断采用的证据：

1. `street`；
2. `overhead`；
3. `gated_mixture`。

`disposition` 表示下一步工作流：

1. `accept`：接受当前风险可控的判断；
2. `acquire_view`：揭示或请求指定 ground-view 方位；
3. `acquire_sensor`：请求 overhead、UAV 或其他传感器；
4. `defer_human`：转人工复核；
5. `field_verify`：升级为现场核查；
6. `stop_low_priority`：在预算约束下暂缓处理。

当 disposition 为 acquisition 时，另存 `acquisition_target`，例如 panorama
sector、building-centered recapture 或 UAV。第一版只需实现三种
`decision_source`，以及 `accept`、`acquire_view` 和 `defer_human`。

### 3.3 输出产品

第一版应输出结构化文件，而不是自由文本：

- `per_sample_decision.csv`：预测、风险、gate weights、`decision_source`、
  `disposition`、`acquisition_target` 和原因；
- `tile_priority.csv`：区域损伤、冲突和复核优先级；
- `inspection_route.geojson`：给定时间或里程预算下的巡检顺序；
- `actionable_report.md`：以上结果的人类可读呈现。

报告是结构化决策的渲染结果，不是核心模型。

## 4. 五个可执行 Workstreams

### Workstream 1: Reliability-Aware Selective Triage

**目标：** 在保持 severe/destroyed recall 的同时，减少必须人工复核的样本数。

第一版可完全复用已有 predictions、visibility features 和 gate features，无需
重训图像 backbone。

基线：

- random review；
- severity probability；
- maximum entropy；
- hard conflict；
- JS divergence；
- calibrated probability average；
- current reliability gate。

指标：

- risk-coverage curve 和 area under risk-coverage curve；
- severe/destroyed Recall@K；
- extreme-error rate；
- selective accuracy；
- 人工工作量减少比例；
- cost-weighted expected loss。

预算应覆盖 5%、10%、20%、30% 和 50%，避免只报告一个人为阈值。

风险控制协议：

1. 将 model fitting、gate fitting、risk calibration 和 final test 分开；若现有
   validation set 太小，应在训练区内建立 nested calibration split；
2. 预先定义有界 loss，例如 severe miss rate、extreme-error cost 或带人工和
   acquisition 成本的复合 loss；
3. 预注册目标风险 `alpha` 和置信水平 `1-delta`；
4. 使用 conformal risk control 或 Learn-then-Test 在 calibration set 上选择
   policy threshold，使风险的单侧上界不超过 `alpha`；
5. 锁定 threshold 后只评价一次 final test，不用 test labels 调整预算或规则；
6. 同时报告 empirical risk、one-sided upper bound、coverage 和 expected cost。

只有满足预注册风险上界的策略才能称为 `risk-controlled`。未满足时应降级为
`risk-aware`，不能仅凭 risk-coverage curve 声称统计风险控制。

Conformal / finite-sample guarantee 依赖 calibration 与 test 的适用条件，通常
包括 exchangeability。In-event spatial holdout 可单独验证 risk-control claim；
zero-shot cross-event transfer 只作为 robustness stress test。若要对新事件继续
声称风险控制，需要目标事件 calibration data 或经过证明的 shift-aware protocol。

### Workstream 2: Active Next-Best-View

**Implementation update (2026-07-10):** the first spatial-v1 CVIAN development
benchmark is complete. Eight overlapping 90-degree sectors, a hidden-view
cache, mask-aware frozen-embedding aggregator, supervised action selector, four
online baselines, four privileged/oracle baselines, five training seeds, and a
descriptive spatial-block bootstrap are implemented. At the recorded `k=3`
development budget, the learned selector does not beat random, farthest angular
coverage, or privileged maximum-building selection, while a privileged greedy
one-step label-aware reference reduces operational cost from about 0.81 to
0.34. That reference is not a globally optimal multi-step acquisition upper
bound. The task has substantial headroom, but the present selector is a
development **No-Go**.

The implementation is reveal-order invariant only under fixed canonical sector
indexing; its raw sector mask is not generally permutation-invariant or
rotation-equivariant. The 80/20 `model_fit`/`selector_fit` split isolates the
downstream heads, but the frozen base checkpoint was trained on the complete
training role, including selector-fit blocks. The test has six spatial blocks
but only four sequence-connected dependency components. A defect correction
required re-evaluating this test during implementation, so it is consumed and
cannot support a later confirmatory selector claim. A future GO requires
spatial-block OOF utility targets or base checkpoints isolated from
selector-fit blocks, plus a new sequence- or event-held-out final test. See
[`docs/results/active_view_protocol.md`](results/active_view_protocol.md) and
[`docs/results/active_view_cvian_spatial_v1.md`](results/active_view_cvian_spatial_v1.md).

**Final execution update (2026-07-11):** the repaired four-role experiment is
also complete and remains a NO-GO. Its primary role is
`selector_selection_holdout_with_historical_base_exposure` (137 samples, 22
sequence components, 21 severe samples), so it improves downstream role
isolation but is not a never-seen external confirmation. At fixed `k=3`, the
registered utility policy is worse than farthest angular coverage by `0.050935`
component-macro cost (farthest-minus-utility 95% CI
`[-0.114026, -0.000229]`) and is not superior to privileged maximum-building
selection (contrast `-0.025325`, 95% CI `[-0.054784, 0.001634]`). A Windows
CRLF verifier defect was independently bounded to newline representation; the
raw hash DAG, semantic fingerprint, result, and both 10,000-draw bootstraps
verify. The resulting claim is **hash-chain-verified consumed
development/exploratory within-CVIAN NO-GO**, not confirmatory evidence. See
[`docs/results/cvian_sequence_utility_v1.md`](results/cvian_sequence_utility_v1.md)
and the
[`integrity addendum`](results/cvian_sequence_utility_v1_integrity_addendum.md).

The frozen Milton sensitivity run is also complete. Relative geometry beats
`random_mc32` descriptively under both dependency-group and spatial-block
aggregation, but its intervals against farthest and clockwise cross zero. It
therefore supplies a bounded sensitivity signal, not a GO, transfer claim, or
external confirmation. The independent post-score verifier passed with
`verified=true`, no defect or downgrade, and independently matched the hash
chain, all decisions/RNG replays, estimands, and bootstraps without inference or
rescoring. See
[`docs/results/active_view_milton_zero_shot_v1.md`](results/active_view_milton_zero_shot_v1.md).
The human-readable integrity record is
[`docs/results/active_view_milton_zero_shot_v1_integrity.md`](results/active_view_milton_zero_shot_v1_integrity.md).
There is no legitimate current-data continuation by tuning these consumed tests.

**目标：** 当前证据不足时，选择最能降低决策风险的下一视角。

Ian 和 Milton 的现有完整 panorama 只能支持离线
**sequential-reveal benchmark**，不能直接证明现实中的数据采集收益。协议为：

1. 预先固定 initial state，例如 overhead 加一个指定初始 sector，或 overhead
   only；
2. 将完整 panorama 隐藏在 evaluation environment 中，并划分为 8 或 12 个
   重叠候选 azimuth sectors；
3. policy 每一步只能访问已经 revealed 的视图和特征，不能读取未揭示 sector
   的 visibility、confidence 或 embedding；
4. 每次 `reveal_sector`、`stop` 和 `defer_human` 都分配预注册 cost；
5. 以逐样本 utility reduction 作为训练目标：
   `loss_before - loss_after - acquisition_cost`；
6. greedy one-step label-aware reference gap 只作为数据集级评价指标，不作为逐样本
   reward，也不能表述为全局 multi-step acquisition upper bound；
7. 真实标签只用于 training/calibration，policy 不得在 test 时读取标签；
8. 在空间隔离或跨事件测试集上评价。

第一阶段采用监督式 action prediction 或 contextual bandit，不需要一开始就用
reinforcement learning。

真正的 next-view acquisition claim 需要额外的多方位现场、车辆或 UAV 数据做
外部验证。在此之前，论文应使用 `sequential evidence reveal` 的准确表述。

基线：

- random azimuth；
- panorama center crop；
- maximum building ratio；
- building-centroid crop；
- maximum confidence；
- maximum expected entropy reduction；
- oracle best action。

指标：

- accuracy / severe recall versus number of revealed views；
- oracle-gap closure per acquired view；
- action regret；
- expected acquisition cost；
- stop/defer calibration；
- leave-one-event-out transfer。

### Workstream 3: Conflict-to-Route

**目标：** 将样本级冲突转成固定时间、里程或停靠点预算下的巡检路线。

输入包括经纬度、severity probability、conflict、gate risk 和 visibility。
输出是区域 priority score 及路线顺序。

关键比较：

- random routing；
- nearest-neighbor routing；
- predicted severity only；
- uncertainty only；
- conflict only；
- reliability-aware priority；
- severity 与 conflict 的联合策略。

指标：

- severe/destroyed discoveries per kilometer；
- severe Recall@travel-budget；
- NDCG@K；
- 首个 severe case 的发现成本；
- route length 和 coverage；
- 不同 grid size、road network 和 minimum tile support 下的稳定性。

“Label-free”应严格写成“目标事件推理阶段无需新增标签”。若 base model 使用了
同一事件的训练标签，就不能宣称整个系统是 end-to-end label-free。最强协议是
在其他事件训练，在完全 held-out event 上生成 priority map。

### Workstream 4: Cross-View Pair Integrity and Alignment QA

**目标：** 判断两个视图是否真正观察同一目标，以及是否应该重新裁图、重新
配对或拒绝 fusion。

可通过受控 corruption 构造训练与评估数据：

- 同 tile 近邻错配；
- 不同 tile 远距离错配；
- 经纬度扰动；
- random panorama azimuth；
- overhead crop shift；
- target building absent / partially visible。

行动输出包括 `accept_pair`、`recrop`、`rematch` 和 `reject_fusion`。

指标包括 mismatch AUROC、Recall@K、top-1 geolocation error、expected
calibration error，以及配对修复对下游损伤判断的增益。

通用 disaster geolocalization 已有 CVDisaster 和 ProbGLC。该 workstream 必须
聚焦 disaster-induced appearance change、pair-quality action 和 downstream
reliability，不能只重复普通 cross-view retrieval。

### Workstream 5: Task-Conditioned Evidence and Grounded Reports

**目标：** 对不同巡检字段学习不同的 view reliability，而不是让一个 gate
同时回答所有问题。

候选字段包括：

- damage severity；
- structure type / category；
- roof construction；
- eaves 和 vent screen；
- exterior siding 和 window pane；
- building visibility / target alignment；
- verification-needed。

每个字段输出：

- predicted value 或 prediction set；
- confidence / calibrated risk；
- supporting view；
- evidence location；
- abstain / verification flag。

报告必须由这些可验证字段生成。自由文本可以作为最后的语言层，但不得替代
字段级 ground truth、evidence grounding 和 decision-utility evaluation。

## 5. 必须先完成的数据与复现修复

### P0.1 CVIAN Georeference and Split Audit

**Implementation update (2026-07-10):** the official position file is now
checksum-verified and joined one-to-one to all local pairs; spatial-block and
sequence-grouped protocols, boundary-buffer provenance, schema tests, and
leakage audits are implemented. The required five-seed performance comparison
is also complete: repaired macro-F1 is lower by 0.049 to 0.097 depending on
mode, confirming that the legacy split overstated generalization. See
[`docs/results/cvian_georeference_split_audit.md`](results/cvian_georeference_split_audit.md)
and [`docs/results/ian_split_performance_comparison.md`](results/ian_split_performance_comparison.md).
The ISPRS and SIGSPATIAL drafts and the CVDisaster comparison note now identify
the official georeference and no longer describe the legacy index grouping as
a clean spatial split.

- 下载并记录官方 CVIAN position 文件的 checksum 和版本；
- 建立 image id 与 position feature 的连接；
- 使用空间 block 或 panorama-location group 重新划分 train/val/test；
- 比较现有 split 与空间 split 的性能差异；
- 修正文稿中“position not released”和 object-grouped split 的表述。

### P0.2 Persist Per-Sample Evidence

**Implementation update (2026-07-10):** `train_reliability_gate.py` now writes
the versioned `p0.2-v1` per-sample evidence schema with dynamic class
probabilities, complete gate weights, visibility/reliability features,
predictions, targets, event/group/location metadata, and a fixed-gate artifact
fingerprint. The repaired five-seed run now emits separate gate-fit,
risk-calibration, and final-test evidence from one frozen gate per seed. See
[`docs/results/per_sample_evidence_schema.md`](results/per_sample_evidence_schema.md).

修改 `train_reliability_gate.py`，逐样本保存：

- street、remote 和 crossview probabilities；
- two-view 与 three-view gate weights；
- visibility features；
- calibrated confidence、entropy 和 JS divergence；
- hard conflict；
- selected prediction 和 target；
- sample id、object id、location 和 tile id。

### P0.3 Temporal and Missing-View Data Loader

**Implementation update (2026-07-10):** the triage dataset now supports
optional pre/post street and overhead views with separate availability,
real-missing, and artificial-dropout masks. The Milton builder now accounts for
all 2,555 source rows, preserves the 254 source-val rows for audit, and provides
a buffered spatial protocol. Collected-versus-generated post-SVI provenance is
still unresolved. Masks are exposed by the loader but are not yet consumed by a
mask-aware fusion model. See
[`docs/results/milton_manifest_split_audit.md`](results/milton_manifest_split_audit.md).

- 支持可选的 pre-street、post-street、pre-overhead、post-overhead；
- 不再要求每种实验都无条件加载两个 post-event views；
- 为缺失视图提供 mask，并明确真实缺失与人为 dropout；
- 核对 Milton source `val` rows 和完整样本计数。

### P0.4 DINS Field Join and Provenance

**Implementation update (2026-07-10):** the official expanded Eaton layer is
now fetched with paginated count verification, metadata/domains/statistics are
preserved, and cross-service `OBJECTID` use is prohibited. A conservative
spatial join matched 19,776/19,780 attachment rows without ambiguity; four
remain in the manual provenance queue. Six non-overwriting derived artifacts
were written beside the local attachment manifest. See
[`docs/results/dins_field_join_provenance.md`](results/dins_field_join_provenance.md).

- 从 CAL FIRE 官方 REST service 获取结构字段；
- 使用稳定键或经过验证的空间连接，不假定不同服务的 `OBJECTID` 一致；
- 记录字段更新时间、空值、`Unknown` 比例和 domain definitions；
- 将 inspector-recorded attributes 与 visually observable evidence 分开；
- 不将 recommended action 或 rationale 伪装成已有 ground truth。

### P0.5 Inaccessible Semantics

**Implementation update (2026-07-10):** the versioned operational ontology
treats Eaton `Inaccessible` as a non-ordinal field-access constraint. It is
allowed only for stress-test/human-escalation use and fails validation if used
as generic uncertainty, an abstention target, a severity class, or a missing-
view marker. See
[`docs/operational_label_cost_ontology.md`](operational_label_cost_ontology.md).

Eaton 的 `Inaccessible` 数量过少，不适合作为主要新分类头；“现场不可进入”也
不等同于“图像看不见目标”。该字段只应用于 stress test、人工升级策略和
业务语义核验，不能直接充当通用 abstention label。

### P0.6 Operational Label and Cost Ontology

**Implementation update (2026-07-10):**
`configs/operational_ontology.json` now preserves every dataset-native label,
defines explicit action-level mappings and directional/resource costs, and
restricts cross-event aggregation to registered event-macro action metrics.
The `wildfire_3class` view is a read-only derived summary that retains its
six-class provenance. See
[`docs/operational_label_cost_ontology.md`](operational_label_cost_ontology.md).

Eaton 的 `no/trace`、`repairable`、`destroyed` 与 Ian/Milton 的
`mild/minor`、`moderate`、`severe` 并非天然等价。跨事件策略应先定义：

- 每个 native label 对应的行动等级；
- adjacent error 与 extreme error 的代价；
- severe miss、false alarm、人工复核和新增视图的相对成本；
- 哪些结果允许跨事件汇总，哪些必须按事件报告。

不能仅凭相同的 class index 将三个数据集拼成统一决策任务。

### P0.7 Risk-Control Split and Claim Rule

**Implementation update (2026-07-10):** a strict three-role protocol now
audits gate-fit/risk-calibration/final-test disjointness, calibrates a fixed
threshold grid with simultaneous finite-sample upper bounds, evaluates the
locked policy once, and downgrades failed or cross-event claims to
`risk-aware`. The CLI writes risk-coverage/AURC, review-budget recall,
calibration diagnostics, and `per_sample_decision.csv`. Formal bounds operate
on registered spatial-group macro losses rather than treating correlated image
rows as independent. On repaired Ian, all five seeds correctly remain
`risk-aware`: severe calibration examples occur in only two blocks, so no 10%
severe-FNR threshold is certifiable. See
[`docs/results/selective_triage_protocol.md`](results/selective_triage_protocol.md)
and [`docs/results/selective_triage_ian_spatial_v1.md`](results/selective_triage_ian_spatial_v1.md).

- 将 gate-fitting data 与 risk-calibration data 分开；
- 预注册 primary loss、`alpha`、`delta` 和 threshold grid；
- 选择适合 loss 单调性的 Conformal Risk Control 或 Learn-then-Test；
- 保存 calibration sample count、selected threshold 和 risk upper bound；
- 分开报告 in-event guarantee 与 zero-shot cross-event stress test；
- 若 held-out risk upper bound 超过目标，只报告 risk-aware result。

## 6. 十二周执行计划

### Weeks 1–2: Data and Protocol Repair

- 完成 P0.1–P0.7；
- 生成统一的 per-sample evidence table；
- 建立 spatial-block 和 leave-one-event-out protocols；
- 增加 split leakage、manifest schema 和 output-column tests。

### Weeks 3–4: Selective Triage and Routing MVP

**Implementation update (2026-07-10):** the selective evaluator, group-aware
risk bounds, fail-closed decisions, seven routing policies, fixed-stop/distance
budgets, GeoJSON route, and actionable Markdown report are implemented. The
Ian seed-42 three-stop case is generated; the statistical go criterion is not
met because the available independent calibration-block count is too small.

- 实现 `scripts/eval_selective_triage.py`；
- 实现 `scripts/build_actionable_report.py`；
- 实现基于 K stops / travel distance 的 route simulation；
- 产出 risk-coverage、Recall@budget 和路线图；
- 判断 conflict 是否在 severity probability 之外提供增量效用。

### Weeks 5–8: Active-View Benchmark

**Final execution update (2026-07-11):** the current-data active-view branch is
complete. The original spatial-v1 and repaired four-role CVIAN experiments both
returned development NO-GO. The four-role test is consumed and carries
historical base exposure; it cannot be recycled into confirmation. The one-time
Milton zero-shot run is complete as sensitivity-only: relative geometry has a
positive descriptive contrast against random, but the primary intervals against
farthest and clockwise cross zero. Adaptive stop/defer remains secondary and
uncalibrated: fail-closed STOP is disabled and the threshold-1 heuristic is only
risk-aware. RQ2/RQ3 cannot proceed on current assets because real field-level
damage, observability/judgment/abstention, and held-out attestation labels do not
exist. The next executable phase is external annotation/data acquisition followed
by a newly frozen, role-isolated protocol—not further tuning on CVIAN or Milton
v1.

- 生成 Ian/Milton multi-azimuth crop manifests；
- 建立 next-view baselines；
- 训练监督式 action selector；
- 完成 action-cost 和 oracle-regret evaluation；
- 进行 Ian-to-Milton 与 Milton-to-Ian transfer。

### Weeks 9–12: External Validation and Paper Package

**Current disposition (2026-07-11):** 现有三事件不能完成 attestability/
attestation-coverage 的外部确认；Milton 已限定为 sensitivity，Eaton 缺损伤构念，
CVIAN 测试已消耗。该阶段在新数据到位前 BLOCKED。

- 取得新的 property-centric event 或真实 multi-azimuth field/vehicle/UAV
  acquisition sequence；
- 取得满足 annotation contract 的 field × view 可证实性标注与 Eaton
  component-damage dominance 参照；
- 在查看新 test outcome 前冻结 task、role isolation、baselines、costs、风险规则
  和 primary metrics；
- 仅在这些 unlock 条件满足后执行 prospective/external evaluation 与 paper
  package。

## 7. Go / No-Go 判据

### Selective Triage

继续主线需要满足：在相同 review budget 下，reliability-aware policy 的
severe/destroyed recall 高于 best non-cross-view baseline，且主要比较的
bootstrap 95% confidence interval 下界高于零。

### Active Next-Best-View

继续主线需要满足：在相同 view budget 下，learned policy 优于
building-centered heuristic，并降低 severe miss 或 extreme-error cost。只提高
平均 accuracy 而不改善风险指标不足以支持 active-evidence claim。

当前 CVIAN 结果是 development No-Go，不是通过预注册 confirmatory test 得到的
最终结论。未来任何 GO 必须使用新的 sequence/event holdout，并在 base encoder
与 action-target 生成层面完成 OOF/role isolation；不得继续使用已消耗的空间 test
选择 selector、预算或 Go 规则。

四角色修复实验没有改变该判定：utility policy 相对 farthest 的主要区间完全位于
零以下，相对 privileged maximum-building 的区间跨零。CRLF 完成验证缺陷已由
独立复核限定在换行表示，但使最终 claim 必须降级为 **hash-chain-verified
consumed development/exploratory within-CVIAN NO-GO**。Milton 只提供 relative
对 random 的正向描述性敏感性；对 farthest/clockwise 未确认，故没有 GO。

这关闭的是**当前数据上的 label-cost selector 分支**，不是对 attestability /
attestation-coverage 中心假说的检验。该假说仍未完成、未确认；在取得字段级
可证实性标注、独立事件/真实多方位序列和新的 prospective test 之前，不得把
label-cost utility 改名为 attestation coverage，也不得继续声称主线已经验证。

### Conflict-to-Route

继续主线需要满足：conflict 或 gate risk 在 severity probability 之外提供可
重复的增量发现率。若 severity-only 始终最优，应将 routing 作为产品输出，而
不是独立科学贡献。

### Grounded Report

仅当结构字段、evidence source 和 verification action 可独立验证时，才进入
VLM report experiment。ROUGE、BLEU 或单一 LLM judge 不能作为主要证据。

## 8. 建议的论文定位

**Working title**

> CrossViewGuard: Risk-Controlled Active Evidence Acquisition for
> Post-Disaster Assessment

**Central claim**

> Cross-view disagreement should trigger an action, not merely another class.

**Current evidence boundary (2026-07-11):** this remains a research motivation,
not an established empirical claim. The current publishable package is a
protocol/benchmark plus development NO-GO, sensitivity, anatomy 0/3, and explicit
data-dependency results. A paper claiming view-conditional attestability or
attestation-coverage superiority requires the external unlock conditions listed
in the consolidated execution status; the present results cannot support that
claim.

**Three contributions**

1. 定义 disaster cross-view selective triage 和 next-best-view task；
2. 提出 visibility- and conflict-conditioned action policy，在固定证据预算下控制
   severe-error risk；
3. 将 sample-level evidence decisions 转成可审计的 review queue、priority map
   和 inspection route，并进行 spatially held-out 和 cross-event evaluation。

当前 ISPRS gate 稿应继续保持 oracle gap、reliability gate、FOV intervention
和 conflict-density map 的清晰叙事。CrossViewGuard 适合作为 follow-up paper，
避免把尚未执行的 agent/report 内容加入现稿并稀释贡献。

## 9. 不建议优先做的方向

1. **只把三类改成六类。** 类别极不平衡，且没有改变科学问题。
2. **通用 VLM 报告生成。** DisasterM3、RAPID、DisasterInsight 和
   GeoDisaster 已覆盖报告与 agentic workflow；没有 reliability、abstention 和
   decision utility 的实现缺乏差异化。
3. **普通 cross-view geolocalization。** CVDisaster 和 ProbGLC 已有直接工作；
   新任务必须聚焦灾后外观变化、pair integrity 或 action-aware localization。
4. **把 `Inaccessible` 直接当作“不确定”。** 业务语义和样本规模均不支持。
5. **先上复杂强化学习。** 应先证明简单监督式策略和启发式基线之间存在稳定
   gap，再增加算法复杂度。

## 10. References and Data Sources

- CAL FIRE Eaton DINS public layer:
  <https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/DINS_2025_Eaton_Public_View/FeatureServer/0>
- CAL FIRE Eaton/Palisades expanded DINS layer:
  <https://services.arcgis.com/HdtZMT2FmI4wPzTM/arcgis/rest/services/Eaton_Palisades_DINS/FeatureServer/2>
- CAL FIRE Palisades DINS public item:
  <https://www.arcgis.com/home/item.html?id=c336759e45764c45861a1e62c4c5e2db>
- CVDisaster code and CVIAN structure:
  <https://github.com/tum-bgd/CVDisaster>
- CVDisaster paper: <https://arxiv.org/abs/2408.06761>
- ProbGLC: <https://arxiv.org/abs/2512.20056>
- DisasterM3: <https://arxiv.org/abs/2505.21089>
- RAPID: <https://arxiv.org/abs/2606.21819>
- GeoDisaster: <https://arxiv.org/abs/2606.17246>
- DisasterInsight: <https://arxiv.org/abs/2601.18493>
- Recov-Vision: <https://arxiv.org/abs/2509.20628>
- Learning to Select Views for Efficient Multi-View Understanding:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Learning_to_Select_Views_for_Efficient_Multi-View_Understanding_CVPR_2024_paper.html>
- Evidential Active Recognition:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Fan_Evidential_Active_Recognition_Intelligent_and_Prudent_Open-World_Embodied_Perception_CVPR_2024_paper.html>
- Conformal Risk Control: <https://arxiv.org/abs/2208.02814>
- Learn then Test: <https://arxiv.org/abs/2110.01052>
