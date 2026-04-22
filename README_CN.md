# CrossViewConflict

这是一个为论文服务的精简仓库，聚焦一个非常具体的问题：

**跨视角融合在什么时候最有价值？**

我们当前的中心论点是：

> `crossview` 的最大价值不在平均样本，而在单视角证据冲突的样本；
> 并且这种价值会随着地面视角类型不同而改变。

这里我们专门比较两种地面视角机制：

- 山火：`building / property-centric` 近建筑视角
- 飓风：`360 / panoramic environment-centric` 环境全景视角

## 为什么单独开这个仓库

我们之前更大的系统项目里还包含：

- 定位
- 生成
- 不确定性
- 更完整的系统框架

而这篇论文需要的是一个更聚焦、更干净的仓库，只围绕：

1. `street_only`
2. `remote_only`
3. `crossview`
4. `conflict subset`
5. 跨数据集、跨视角机制比较

## 数据集说明

数据太大，不放仓库，只保留本地。

本仓库假定你本机已经有：

- `Altadena / Eaton wildfire paired dataset`
- `IAN_hurricane`

我们机器上的示例路径：

- `C:/Users/yyang295/Desktop/Altadena_Images`
- `C:/Users/yyang295/Desktop/IAN_hurricane`

## 仓库包含什么

- 数据 manifest 构建脚本
- triage 训练脚本
- triage 评估脚本
- conflict subset 构建脚本
- 论文导向的结论文档

核心脚本：

- `scripts/build_eaton_manifest.py`
- `scripts/build_ian_hurricane_manifests.py`
- `scripts/train_triage.py`
- `scripts/eval_triage.py`
- `scripts/build_conflict_subset.py`

## 任务定义

### 山火

沿用我们在 Eaton / Altadena 山火实验里已经固定好的二分类映射：

- `0 = No Damage + Affected`
- `1 = Minor + Major + Destroyed`
- `Inaccessible` 不参与

### 飓风

`IAN_hurricane` 原始是三类：

- `0_MinorDamage`
- `1_ModerateDamage`
- `2_SevereDamage`

本仓库只保留最轻和最重两类，做成更干净的端点二分类：

- `0 = MinorDamage`
- `1 = SevereDamage`

中间类 `ModerateDamage` 暂时去掉。

## 目前最重要的结果

### 山火 building-view

来自前期山火实验结果：

- `crossview` test `F1 = 0.9713`
- `street_only` test `F1 = 0.9604`
- `remote_only` test `F1 = 0.9653`

在冲突子集上：

- `street_only = 0.4179`
- `remote_only = 0.5821`
- `crossview = 0.7612`

### 飓风 360-view

本仓库中已经完成的对比实验：

- `crossview` test `F1 = 0.9208`
- `street_only` test `F1 = 0.8912`
- `remote_only` test `F1 = 0.9082`

在冲突子集上：

- `street_only = 0.4286`
- `remote_only = 0.5714`
- `crossview = 0.6190`

## 当前结论

有两层结论已经比较清楚了。

第一层：
`crossview` 在两种灾种、两种地面视角机制下都还是最好的。

第二层，更有论文价值：

- `crossview` 的价值确实主要体现在 `conflict subset`
- 但这种提升在山火 `building-view` 上更强
- 在飓风 `360-view` 上虽然也存在，但幅度更温和

这支持一个很好的论文立意：

**cross-view fusion 的收益和 ground-view 与目标建筑的对齐程度有关。**

也就是：

- 地面图越“对准建筑本体”，crossview 越强
- 地面图越“偏环境全景”，crossview 依然有用，但增益会变小

## 轻量 alignment 分析

我还做了一版轻量的语义分割分析，用冻结的 `SegFormer` 估计冲突子集地面图里
`building` 像素占比和建筑质心相对图像中心的位置。

结果很清楚：

- 山火 conflict 图像的平均 `building_ratio = 0.2684`
- 飓风 conflict 图像的平均 `building_ratio = 0.0154`
- 山火图像里的建筑区域也更靠近图像中心

这说明“山火上的 crossview 增益更强”不再只是一个观察，而是已经有了可解释的
`target alignment proxy` 来支持。

## 新增的统计与补充实验

我又补了四类 reviewer 很在意的实验：

1. `Permutation test`
   - 山火 conflict 上结果明显强于 label-independence null
   - 飓风目前还是同方向趋势，但样本太小，暂时不适合写成强显著结论
2. `Threshold sensitivity`
   - 在不同 soft-conflict 阈值下，crossview 都仍然是最好的
3. `Per-sample correlation`
   - 当前 `building_ratio` 和单样本的 crossview 是否纠错之间，没有出现很强的单调相关
   - 这说明 alignment 更像是一个 `dataset / view-regime` 层面的机制，而不是当前 conflict 子集内的细粒度排序信号
4. `Qualitative figure`
   - 现在已经有论文可直接用的冲突案例图

## 论文主线建议

这篇论文最适合写成：

**When Cross-View Helps Most**

而不是简单写成：

**Cross-View Works**

因为你现在真正的研究价值是：

- 什么时候最有帮助
- 为什么在冲突样本上更有帮助
- 为什么不同地面视角机制下帮助强度不同

## 安装与运行

```powershell
cd "C:\Users\yyang295\Documents\New project\CrossViewConflict"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

后续运行方式见英文 README 和 `docs/` 下文档。
