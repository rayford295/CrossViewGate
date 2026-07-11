# CrossViewGate（中文说明）

**看得见目标的视图才值得信任：面向跨视图灾害损伤评估的可见度条件化可靠性门控。**

英文完整说明见 `README.md`，论文稿见 `paper/isprs_manuscript.md`。

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

## 复现

见 `README.md` 的 Reproducing 一节；结果文档索引在 `docs/results/README.md`
（`_v2` 后缀 = 收敛 5-seed 主协议）。

## 更名说明

本仓库原名 `CrossViewConflict`，旧链接自动重定向；Python 包导入名保持
`crossview_conflict` 不变。
