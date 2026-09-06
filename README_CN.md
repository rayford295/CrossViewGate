# CrossViewGate（中文说明）

**看得见目标的视图才值得信任：面向跨视图灾害损伤评估的可见度条件化可靠性门控。**

英文完整说明见 `README.md`，论文完整稿见 `paper/isprs_manuscript.md`，GeoSearch 2026 四页短文见 `paper/geosearch2026_short/`。

## 核心发现

1. **Oracle gap（动机）**：在街景模型与遥感模型预测冲突的样本上，逐样本选对视图的
   oracle 比现有一切融合方法高 0.37–0.41 的准确率（三个灾害数据集一致）——对称融合
   浪费了大部分"该信哪个视图"的信息。
2. **可靠性门控（方法）**：一个**线性**门控（建筑可见度 + 校准置信度 + 跨视图分歧，
   共 11 个可解释特征）逐样本混合 street/remote/crossview 概率。它是唯一显著优于
   校准概率平均的方法（野火冲突子集 +0.051，p=0.0001），并在全测试集上也显著优于
   端到端融合；全景数据集上保持持平、从不显著变差。
3. **因果机制（干预实验）**：把飓风全景裁剪成建筑居中的 90° 视野后融合收益翻倍，
   几何完全相同的随机裁剪则不会（6/6 方向一致）——决定融合价值的是街景的目标对齐，
   而非灾种。
4. **免标注损伤地图（应用）**：跨视图冲突的空间密度无需任何标注即可预测 tile 级损伤
   （野火 Spearman r=0.615，p=0.001），而单视图不确定性密度做不到。
5. **负结果**：单视图模型收敛并校准后，简单概率平均即可打平端到端学习融合——
   学习的价值在于非对称的可靠性仲裁，而不在融合本身。
6. **Active-view 开发结果**：CVIAN 的八扇区离线 sequential-reveal 实验为 No-Go；
   当前 learned selector 未优于简单覆盖或 privileged building heuristic。现有
   model/selector 划分只隔离 downstream heads，空间 test 也已被开发过程消耗。
   未来正结论需要 base-encoder/utility-target OOF 隔离，以及新的 sequence- 或
   event-held-out confirmatory test。当前 label-aware oracle 只是 greedy one-step
   privileged reference，不是全局 acquisition upper bound。
7. **方向修订（2026-07-10）**：研究对象从"拟合单一 severity 标签"转向
   **视角条件可证实性**——标签被视为多面证据的有损投影，跨视图分歧是待解释的
   现象，acquisition 效用从 label-loss 改为证据覆盖增益。这同时给 active-view
   No-Go 提供了对立假说（目标函数错位 vs selector 不成熟）。RQ1 Study A
   （CVIAN 方向性解剖，exploratory，仅用非 test 角色）已执行：三个开发判据均
   未达标——现有可见度特征不携带 held-out 可证实性信号；5/5 种子出现一致但
   未确认的"街景报更重"方向倾斜。Eaton 部件级 Study B 成为关键检验。详见
   [`docs/2026-07-10_crossviewguard_attestability_revision.md`](docs/2026-07-10_crossviewguard_attestability_revision.md)
   与 [`docs/results/disagreement_anatomy_cvian_v1.md`](docs/results/disagreement_anatomy_cvian_v1.md)。

## 复现

见 `README.md` 的 Reproducing 一节；结果文档索引在 `docs/results/README.md`
（`_v2` 后缀 = 收敛 5-seed 主协议）。

## 更名说明

本仓库原名 `CrossViewConflict`，旧链接自动重定向；Python 包导入名保持
`crossview_conflict` 不变。
