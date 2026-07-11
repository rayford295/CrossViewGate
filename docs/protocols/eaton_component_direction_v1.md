# Eaton damaged-component observability：方向性分歧实验协议 v1

**状态（2026-07-11）：** 空间角色、模型与确认集承诺均已冻结；`spatial_confirmation` 仍为未评分、未解盲状态。开发角色在人工标注前触发 `STOP_PRE_ANNOTATION_STRUCTURAL_FUTILITY`：全部 development 只有 28 个 blocks，而注册门槛要求至少 30 个 eligible blocks；五种子 ensemble 也只有 97 个无符号分歧、覆盖 24 blocks，低于 250/30 门槛。因此 v1 不请求人工标注，也不报告 H-B1/H-B2。正式结果见 [`../results/eaton_component_observability_v1.md`](../results/eaton_component_observability_v1.md)。

## 唯一研究问题

在 Eaton 火灾中，当街景与 overhead 图像的三档 ordinal severity 预测不一致时，**受损部件主导性及该部件在两个视角中的可评估性，是否与分歧的正负方向存在系统关联？**

预注册机制预期只有两条：

- roof-dominant 且屋顶只在 overhead 中可评估时，overhead 应比 street 预测更严重；
- facade-dominant 且立面只在 street 中可评估时，street 应比 overhead 预测更严重。

这是同一事件内的空间关联与图像证据解释，不是因果效应、现场真值因果归因、跨事件外推、下一视角选择、路由、stop/defer 或可部署 selector 的证据。若协议成功，允许的中心表述也仅为：**Eaton 的跨视角严重度分歧与受损部件的视角可观测性呈预注册的方向性关联。**

冻结配置见 [`configs/eaton_component_direction_v1.json`](../../configs/eaton_component_direction_v1.json)。
原始配置、summary、四角色 manifests 与媒体 ledger 的外部哈希锚见 [`configs/eaton_component_direction_v1_commitments.json`](../../configs/eaton_component_direction_v1_commitments.json)；正式模型输出的 prediction、metadata、checkpoint 与执行代码哈希见 [`configs/eaton_component_direction_v1_development_run_commitments.json`](../../configs/eaton_component_direction_v1_development_run_commitments.json)。

## 分析队列与空间隔离

分析单位是每个 DINS structure 的一组 media-only street–overhead 图像。首先只保留两幅媒体均可读、坐标有效且有 ordinal source label 的记录；`Inaccessible` 是现场可进入性约束，不纳入 ordinal severity。三档顺序固定为：

1. `no_or_trace_damage`：`No Damage` 或 `Affected (1-9%)`；
2. `damaged_repairable`：`Minor (10-25%)` 或 `Major (26-50%)`；
3. `destroyed`：`Destroyed (>50%)`。

同一 DINS structure 若有多个 attachment，只取 street 图像像素面积最大的一组；再以最小 numeric `attachment_id` 和 lexical `pair_id` 依次打破并列。常量图像、不可读/缺失媒体与 EXIF Orientation≠1 的图像均在任何训练前排除，避免空白证据以及模型和人工查看器看到不同旋转。由 19,780 条 joined rows 得到 18,291 个 canonical structures；1,489 条 source exclusions 逐行保存在冻结审计文件中。

空间协议使用 `0.002°` grid、seed `20260711`，先把共享 exact street/overhead SHA-256 的空间块合并为不可拆分 dependency component，再分配角色并执行跨角色 `50 m` buffer。buffer 共排除 3,234 个 canonical pairs，最终保留 15,057 个：

| Protocol role | Pairs / structures | Spatial blocks | 允许用途 |
| --- | ---: | ---: | --- |
| `model_fit` | 6,755 | 349 | 仅训练单视角模型 |
| `model_validation` | 1,284 | 30 | early stopping、temperature calibration |
| `study_development` | 1,534 | 28 | 开发期预测、盲标注流程与预解盲 gate |
| `spatial_confirmation` | 5,484 | 159 | 一次性同事件空间确认；当前禁止评分 |

冻结的 pairwise audit 在任意两角色之间均为：`pair_id` overlap = 0、DINS structure overlap = 0、spatial block overlap = 0、street/overhead exact-media SHA overlap = 0、同一 orthomosaic tile 上实际 512×512 crop overlap = 0。所有跨角色最近距离严格大于 50 m；全审计中的最小值约为 50.004 m，同 tile crop 中心最小距离约 669.0 pixels。角色比例是在 buffer 前指定，因此上表的最终比例不要求精确等于配置中的 0.5/0.1/0.1/0.3。

确认清单有 5,484 行、159 个 blocks，其 SHA-256 承诺为：

```text
3f7cd19f6afe55bbb761c7a44e6e6dfeea31744fe84d0665c597388629cdcbfe
```

冻结产物及完整距离审计位于 [`data/splits/eaton_component_direction_v1/`](../../data/splits/eaton_component_direction_v1/)。该确认集是 **same-event spatial confirmation**，不是外部事件确认。

## 预测生成与 ensemble 单位

街景与 overhead 分别训练一个 `ResNet-18` 单视角三分类模型。协议固定 image size 224、15 epochs、patience 4、balanced class weighting、learning rate `1e-4`，使用 seeds `42, 123, 456, 789, 1011`。

每个 seed、每个视角的 temperature 只能在 `model_validation` 上拟合。对一个 pair，先对同一视角五个 seeds 的 temperature-calibrated class probabilities 求均值，再以固定 class order 做 `argmax`。此后每个 pair 只有一个 street prediction 与一个 overhead prediction。

**模型 seed 不是独立观察。** 不得把五个 seed rows 当作五个样本扩大样本量、block 数或显著性。推断单位始终是 seed-ensemble 后的 structure/pair，并以 spatial block 重采样。

旧 `altadena_3class` 实验中确实存在 Eaton/Altadena per-view predictions；因此“完全没有 Eaton predictions”的旧表述不准确。但这些预测来自已消费的 legacy、非本协议空间隔离的 pilot/analysis splits，不能替代本协议的 fresh role-isolated predictions，也不能用于注册检验或确认决策。它们最多用于无主张的代码 sanity check 或 pilot yield 估计。

## 人工标注构念

只对指定角色内 **ensemble 发生 street–overhead disagreement 的全部唯一 pairs** 生成标注包；抽样可以使用“是否分歧”，不得使用 signed direction。标注者只看到 opaque `pair_id`、两幅媒体、媒体 SHA-256、协议标识与待填字段，不得看到：

- native/derived severity label；
- 任一视角的 prediction、probability 或 confidence；
- signed disagreement direction；
- spatial block 或 protocol role；
- 另一名标注者的答案。

每个 pair 的 `component_dominance` 取以下一个值：

| 值 | 判定规则 |
| --- | --- |
| `roof` | 两幅图像合看，能够辨认的损伤证据主要由屋顶/屋面构件主导 |
| `facade` | 能够辨认的损伤证据主要由外墙、门窗等垂直立面构件主导 |
| `mixed` | roof 与 facade 均有实质损伤证据，无法合理指定单一主导部件 |
| `none_detected` | 在当前图像证据中未发现可辨认的 roof/facade 损伤；不等于现场无损伤 |
| `unknown` | 因覆盖、遮挡、尺度或质量无法判断哪类受损部件主导 |

还要分别填写 `street_roof_assessability`、`street_facade_assessability`、`overhead_roof_assessability` 与 `overhead_facade_assessability`：

- `assessable`：该视角对该部件有足够覆盖与质量，可以判断其损伤状态；
- `not_assessable`：部件未入镜、被遮挡、像素过小或质量不足，无法判断；
- `indeterminate`：位于前两者边界，标注者不能可靠决定。

“可评估”不是“有损伤”，“不可评估”也不是“无损伤”。DINS construction/material/presence/exposure 字段以及 `dins_wherefirestartedonstructure` 均不得充当 damage dominance；VLM 或其他模型生成的 component labels 也不得充当人工 reference。

整包由固定的两名 raters 独立完成。任一注册字段不一致时必须由第三方 adjudicator 处理；最终 provenance-bearing reference 必须保留两份原始评分、adjudication 记录、protocol hash 与两幅媒体 hash，不能通过 inner join 静默丢弃 pair。

## 机制资格、H-B1 与 H-B2

令

```text
Y_i = 1[overhead_prediction_i > street_prediction_i]
```

仅在 ensemble predictions 不一致时进入方向分析。机制资格在解盲前固定为：

- `roof` stratum：`component_dominance=roof`，且 `overhead_roof_assessability=assessable`、`street_roof_assessability=not_assessable`；
- `facade` stratum：`component_dominance=facade`，且 `street_facade_assessability=assessable`、`overhead_facade_assessability=not_assessable`。

`mixed`、`none_detected`、`unknown` 及所需 assessability 为 `indeterminate` 的记录不进入主分析。

### H-B1（primary）

主估计量为 severity-standardized directional contrast：

```text
Delta_std = P_std(Y=1 | mechanism-eligible roof disagreement)
          - P_std(Y=1 | mechanism-eligible facade disagreement)
```

两个 stratum 均标准化到所有 mechanism-eligible disagreements 的冻结三档 target 分布。区间使用 whole-spatial-block bootstrap，10,000 次，seed `20260711`。若某个有正权重的 severity stratum 缺少 roof 或 facade common support，则主估计不可解释为已通过。

### H-B2（supportive）

H-B2 是机制资格分歧中的 direction-consistent share：roof 中 `Y=1`，或 facade 中 `Y=0`。另外报告：

- mechanism eligibility coverage / all disagreements；
- full-disagreement explained fraction = direction-consistent eligible disagreements / all disagreements。

H-B2、coverage 或 explained fraction 均为支持性结果，**不能补救 H-B1 失败**。

## 解盲前 gate

在查看 signed-direction 统计量前，以下条件必须全部满足：

- 每个注册标注字段 raw agreement ≥ 0.75 且 Cohen's kappa ≥ 0.67；
- mechanism-eligible disagreements ≥ 250；
- roof-eligible ≥ 75，facade-eligible ≥ 75；
- eligible spatial blocks ≥ 30；roof 与 facade 各 ≥ 15 blocks；
- 对最小有意义效应 `Delta_std = 0.20` 的冻结模拟功效 ≥ 0.80。

任一项失败即记录 `STOP_UNDERPOWERED_OR_UNRELIABLE_CONSTRUCT`，不得继续查看或解释 H-B1/H-B2，也不得把停止解释成“机制不存在”。开发角色用于验证标注可靠性、yield、功效与完整流程；在模型、ensemble、codebook、gate、estimator 和所有 hashes 均冻结前，确认角色不得评分或按方向标注。

## 一次性确认决策

只有通过预解盲 gate 才能在 `spatial_confirmation` 上执行一次注册分析。`GO_FOR_REGISTERED_EATON_IMAGE_MECHANISM` 要求以下条件全部成立：

1. `Delta_std >= 0.20`；
2. H-B1 的 95% spatial-block bootstrap lower bound > 0；
3. roof positive-direction share > 0.5；
4. facade positive-direction share < 0.5；
5. mechanism eligibility coverage ≥ 0.30。

否则结论固定为 `NO_GO_FOR_REGISTERED_EATON_IMAGE_MECHANISM`。支持性估计量不能改变该判定。无论 GO 或 NO-GO，都只适用于 Eaton 同一事件的冻结空间确认队列。

## 当前合法边界

空间清单、媒体 hash ledger、角色隔离、fresh development predictions 与确认 hash 均已就绪，但它们不是机制结果。正式模型由 [`scripts/run_eaton_component_models.py`](../../scripts/run_eaton_component_models.py) 生成；无方向容量门禁由 [`scripts/run_eaton_preannotation_futility_gate.py`](../../scripts/run_eaton_preannotation_futility_gate.py) 执行；注册统计实现位于 [`crossview_conflict/analysis/eaton_component_direction.py`](../../crossview_conflict/analysis/eaton_component_direction.py)。

v1 已在人工标注前合法停止，因为任何标注都不能使 28 个全部 blocks 达到 30 个 eligible blocks，且 97 个全部分歧也不能达到 250 个 mechanism-eligible 分歧。因此 H-B1/H-B2 没有合法可报告的数值，confirmation 不得访问。不得以 DINS proxy、legacy predictions、VLM labels 或合成标注越过这一边界；当前状态是**结构性 underpowered STOP**，不是实验的正面或负面机制结果。
