# P0.4 DINS Field Join and Provenance

状态：已实现、完成官方服务 dry run，并在显式 `--write` 下生成了新的派生
artifact 目录；原 manifest 未被原地修改或覆盖。

## 数据源与快照边界

字段源使用 CAL FIRE 发布的 [Eaton/Palisades expanded DINS layer](https://services.arcgis.com/HdtZMT2FmI4wPzTM/arcgis/rest/services/Eaton_Palisades_DINS/FeatureServer/2)，查询条件固定记录为 `INCIDENTNAME = 'Eaton'`。现有附件 manifest 的 URL 来自另一个 CAL FIRE [Eaton public attachment layer](https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/DINS_2025_Eaton_Public_View/FeatureServer/0)。两者是不同的 ArcGIS 服务。

2026-07-10 dry run 读取到：

| 项目 | Expanded field layer | Public attachment layer |
| --- | ---: | ---: |
| Eaton records | 18,422 | 18,428 |
| 官方 metadata `lastEditDate` | 2025-02-20 20:13:40.705 UTC | 2025-06-23 19:24:09.658 UTC |
| 字段数 | 44 | 4 |
| 直接发布 coded-value domain 的字段数 | 0 | 2 |

Expanded layer 没有为 44 个字段发布 field-level coded-value domain；程序会如实保存 `domain: null`，不会把当前观测值自行升级为“官方 domain”。Public layer 为 `DAMAGE` 和 `STRUCTURETYPE` 发布了 coded-value domain，分别包含 6 和 16 个 code；这份参考 metadata 也会单独保留。

## 连接策略

当前 Eaton manifest 只有 public-layer `objectid`、attachment id、类别和坐标，没有与 expanded layer 共享的稳定业务键。因此本次 dry run 使用显式空间连接，绝不比较两个服务的 `OBJECTID`。

程序按以下顺序尝试唯一、非空的稳定键：

1. `GLOBALID`；
2. `INCIDENTNUM + APN + STRUCTURETYPE`；
3. `APN + STRUCTURETYPE`；
4. `SITEADDRESS + STRUCTURETYPE`。

只有源端恰好一个候选时，业务键才成立。`OBJECTID`、`OID` 和 `FID` 即使由命令行显式指定也会被拒绝。没有唯一业务键时才使用 haversine 最近邻：默认容差 10 m，歧义边界 1 m；如果次近候选与最近候选相差不超过 1 m，该行标记为 `spatial_ambiguous`，且不填充 DINS 字段。

每行输出以下审计信息：join status/method/key、最近和次近距离、容差、歧义边界、候选数、业务键歧义、距离告警、字段服务 URL 和服务更新时间。Expanded-layer `OBJECTID` 只保留为 `dins_source_objectid` provenance；snapshot 另存 canonical-JSON SHA-256，输入 manifest 另存文件 SHA-256。

## 官方数据 dry-run 结果

输入是本地 `dataset_index.csv` 的 19,780 个 attachment rows。官方查询按 `OBJECTID ASC` 以 2,000 条每页抓取，共 10 页；预查询 count 与最终 GeoJSON feature count 均为 18,422。

| 指标 | 结果 |
| --- | ---: |
| Manifest attachment rows | 19,780 |
| 无歧义匹配 rows | 19,776 (99.9798%) |
| 未匹配 rows | 4 |
| 空间歧义 rows | 0 |
| 被 attachment 使用的唯一 expanded records | 18,411 |
| 没有对应 attachment 的 expanded records | 11 |
| 对应多个 attachment rows 的 expanded records | 1,177 |
| 匹配距离 p50 / p95 / p99 | <0.001 m / <0.001 m / <0.001 m |
| 最大匹配距离 | 3.992 m |

4 个未匹配行保持空值，不通过放宽容差或假设同号 `OBJECTID` 强行连接。未来正式写入前应把它们作为人工 provenance queue 检查。

同日的显式写入运行复现了完全相同的 count、join status 和距离统计，并在本地
`Eaton_Fire_attachments_index_output/dins_field_join/` 下保存六项派生产物：官方
GeoJSON 快照、完整 service metadata、field domains、field statistics、
`eaton_manifest_with_dins.csv` 和 `join_provenance.json`。这次运行未使用
`--overwrite`，也未更改输入 `dataset_index.csv`。

部分字段质量统计说明为什么空值和 `Unknown` 必须分开报告：

| 字段 | Null/blank | `Unknown` / all rows | 语义 |
| --- | ---: | ---: | --- |
| `EAVES` | 0.00% | 37.40% | visually observable candidate |
| `WINDOWPANE` | 0.00% | 29.84% | visually observable candidate |
| `VENTSCREEN` | 0.00% | 27.66% | visually observable candidate |
| `PATIOCOVERCARPORT` | 0.00% | 13.57% | visually observable candidate |
| `DEFENSIVEACTIONS` | 58.87% | 37.25% | inspector assessment/observation |
| `WHEREFIRESTARTEDONSTRUCTURE` | 95.34% | 0.51% | conditional inspector observation |
| `WHATDIDFIRESTARTFROM` | 95.34% | 0.48% | conditional inspector observation |
| `BATTALION` / `FIRENAME` | 100.00% | 0.00% | administrative |

统计定义固定为：JSON null、NaN、空字符串和纯空白属于 null/blank；去除首尾空白后大小写不敏感、精确等于 `Unknown` 才计为 Unknown。

## 语义边界

所有新增值都源于 DINS inspector/service record。代码将字段分成：

- `visually_observable_candidate`：结构类型、屋顶、檐口、通风网、外墙、窗、露台等；只有相关部位在给定图像中可见时才可用于视觉事实评估；
- `inspector_assessment_or_observation`：damage、起火位置/来源、防御行为和数量字段；
- `administrative_or_parcel`：事件、地址、APN、估值、建造年份和位置字段；
- `source_identifier`：ArcGIS source identifiers。

“visually observable candidate” 不是 image-level visibility ground truth。流程不会创建 `recommended_action` 或 `rationale` 字段，也不会把它们描述成已有 ground truth。

## 使用方式

默认命令只读官方服务并打印摘要，不创建输出目录：

```powershell
python scripts/fetch_dins_fields.py `
  --manifest-csv C:\path\to\dataset_index.csv
```

完全离线重放使用本地 GeoJSON 和 metadata；此模式不会发起任何网络请求：

```powershell
python scripts/fetch_dins_fields.py `
  --manifest-csv C:\path\to\dataset_index.csv `
  --source-geojson C:\snapshot\dins_fields.geojson `
  --source-metadata C:\snapshot\service_metadata.json `
  --reference-metadata C:\snapshot\public_service_metadata.json
```

只有显式加入 `--write` 才会写入一个新的派生目录；已有 artifact 默认拒绝覆盖，必须再显式加入 `--overwrite`：

```powershell
python scripts/fetch_dins_fields.py `
  --manifest-csv C:\path\to\dataset_index.csv `
  --output-dir data\dins\eaton_field_join `
  --write
```

写入模式生成：`dins_fields.geojson`、完整 `service_metadata.json`、`field_domains.json`、`field_statistics.json`、派生的 `eaton_manifest_with_dins.csv` 和 `join_provenance.json`。原 manifest 永不原地修改。

## 验证

```text
python -m pytest tests/test_dins_join.py -q
.......                                                                  [100%]
7 passed
```

测试完全使用本地夹具，覆盖稳定键优先、同号 `OBJECTID` 不连接、空间距离和歧义、空值与 Unknown、分页 count 完整性、离线 CLI dry run，以及默认拒绝覆盖。
