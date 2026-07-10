# CrossViewGuard: 从灾害分类到风险可控的主动证据获取

- **Date:** 2026-07-10
- **Status:** research roadmap, not yet executed
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
- CAL FIRE 的公开 DINS 服务还包含 structure type、structure category、roof、
  eaves、vent screen、exterior siding、window pane、deck/porch 等结构字段。

这些 DINS 结构字段目前没有进入 CrossViewGate manifest。它们需要从官方服务
重新获取并通过稳定的空间或记录键连接，不能把当前 damage category 误写成
完整的 report ground truth。

### 2.2 Hurricane Ian / CVIAN

- 4,121 个 post-disaster street/satellite pairs；
- 当前本地 CSV 只包含图像路径和 severity；
- CVDisaster 官方完整数据包另含
  `02_Position/CVIAN_position.geojson` 和 position shapefile。

这意味着 Ian 并非无法进行空间分析。应导入官方 position 文件，再建立空间
block split、conflict map 和 route simulation。当前 builder 给每行分配顺序
`objectid`，不能证明论文所称的 object-grouped split。

### 2.3 Hurricane Milton

- 源数据包含 pre-disaster street、post-disaster street、post-disaster
  satellite、经纬度、prompt 和 source split；
- 当前 manifest 已保留这些字段；
- 当前 `CrossViewTriageDataset` 只读取 post street、post satellite、label 和
  sample id，未使用 pre-disaster street；
- GenDisasterSVI 数据清单记录 2,555 个 indexed samples，并单独列出
  train/val/test；它不同于 `Bi-temporal_hurricane` 清单中的 2,556 个 paired
  folders。当前 builder 只选择 source `train` 和 `test`，需要直接审计源 CSV
  的 `set` 分布、过滤条件和最终保留行数。
- 数据名称和 manifest 中保留的 `prompt` 表明还需核验 post-SVI 的生成与采集
  provenance。若部分图像为生成数据，Milton 应作为敏感性分析，真实 CVIAN
  应作为 active-view 主结果。

Milton 因此适合研究“视角乘以时间”的证据增益，而不是继续把信息压缩成一个
三分类输入。

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
6. oracle-gap closure 只作为数据集级评价指标，不作为逐样本 reward；
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

- 下载并记录官方 CVIAN position 文件的 checksum 和版本；
- 建立 image id 与 position feature 的连接；
- 使用空间 block 或 panorama-location group 重新划分 train/val/test；
- 比较现有 split 与空间 split 的性能差异；
- 修正文稿中“position not released”和 object-grouped split 的表述。

### P0.2 Persist Per-Sample Evidence

修改 `train_reliability_gate.py`，逐样本保存：

- street、remote 和 crossview probabilities；
- two-view 与 three-view gate weights；
- visibility features；
- calibrated confidence、entropy 和 JS divergence；
- hard conflict；
- selected prediction 和 target；
- sample id、object id、location 和 tile id。

### P0.3 Temporal and Missing-View Data Loader

- 支持可选的 pre-street、post-street、pre-overhead、post-overhead；
- 不再要求每种实验都无条件加载两个 post-event views；
- 为缺失视图提供 mask，并明确真实缺失与人为 dropout；
- 核对 Milton source `val` rows 和完整样本计数。

### P0.4 DINS Field Join and Provenance

- 从 CAL FIRE 官方 REST service 获取结构字段；
- 使用稳定键或经过验证的空间连接，不假定不同服务的 `OBJECTID` 一致；
- 记录字段更新时间、空值、`Unknown` 比例和 domain definitions；
- 将 inspector-recorded attributes 与 visually observable evidence 分开；
- 不将 recommended action 或 rationale 伪装成已有 ground truth。

### P0.5 Inaccessible Semantics

Eaton 的 `Inaccessible` 数量过少，不适合作为主要新分类头；“现场不可进入”也
不等同于“图像看不见目标”。该字段只应用于 stress test、人工升级策略和
业务语义核验，不能直接充当通用 abstention label。

### P0.6 Operational Label and Cost Ontology

Eaton 的 `no/trace`、`repairable`、`destroyed` 与 Ian/Milton 的
`mild/minor`、`moderate`、`severe` 并非天然等价。跨事件策略应先定义：

- 每个 native label 对应的行动等级；
- adjacent error 与 extreme error 的代价；
- severe miss、false alarm、人工复核和新增视图的相对成本；
- 哪些结果允许跨事件汇总，哪些必须按事件报告。

不能仅凭相同的 class index 将三个数据集拼成统一决策任务。

### P0.7 Risk-Control Split and Claim Rule

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

- 实现 `scripts/eval_selective_triage.py`；
- 实现 `scripts/build_actionable_report.py`；
- 实现基于 K stops / travel distance 的 route simulation；
- 产出 risk-coverage、Recall@budget 和路线图；
- 判断 conflict 是否在 severity probability 之外提供增量效用。

### Weeks 5–8: Active-View Benchmark

- 生成 Ian/Milton multi-azimuth crop manifests；
- 建立 next-view baselines；
- 训练监督式 action selector；
- 完成 action-cost 和 oracle-regret evaluation；
- 进行 Ian-to-Milton 与 Milton-to-Ian transfer。

### Weeks 9–12: External Validation and Paper Package

- 优先加入 Palisades DINS 作为第二个 property-centric wildfire event；
- 或在数据权限不完整时，用现有三事件完成 leave-one-event-out validation；
- 锁定 task definition、baselines 和 primary metrics；
- 生成方法图、risk-coverage 图、budgeted discovery 图和路线案例；
- 完成论文 outline 与 reproducibility checklist。

## 7. Go / No-Go 判据

### Selective Triage

继续主线需要满足：在相同 review budget 下，reliability-aware policy 的
severe/destroyed recall 高于 best non-cross-view baseline，且主要比较的
bootstrap 95% confidence interval 下界高于零。

### Active Next-Best-View

继续主线需要满足：在相同 view budget 下，learned policy 优于
building-centered heuristic，并降低 severe miss 或 extreme-error cost。只提高
平均 accuracy 而不改善风险指标不足以支持 active-evidence claim。

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
