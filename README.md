# Virtual Embryo Challenge — Task 1 / Task 2

NeurIPS 2026 [Virtual Embryo Challenge](https://virtualembryo.ai/challenge) 代码。Task 1：由 E8.5 / E9.5 全转录组预测未来阶段的细胞表达分布。Task 2：3D MERFISH 的时空预测（表达 + `spatial_3D`），胚胎与心脏两个 setting 分开建模。

详细设计见 [doc/t1_technical_plan.md](doc/t1_technical_plan.md)、[doc/t2_technical_plan.md](doc/t2_technical_plan.md)，三项任务总计划见 [doc/model_training_plan.md](doc/model_training_plan.md)。T2 OT-CFM 本地评估见 [doc/t2_flow_eval.md](doc/t2_flow_eval.md)。

所有命令都在**项目根目录**执行。原始 `.h5ad` 放在 `data/`。

---

# Task 1

## 1. 环境

Python 3.10–3.13。建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-t1.txt
```

依赖包括 `anndata`、`torch`、`veckit` 等。GPU 可选；没有 GPU 时训练会走 CPU / MPS。

从官网下载基因 panel，放到 `panels/T1__val.genes.txt`（一行一个基因，顺序必须与榜单一致）。若该文件不存在，脚本会退回使用训练文件的基因顺序，并写到 `outputs/t1/gene_order.txt`。

---

## 2. 需要的数据

| 文件 | 用途 |
|------|------|
| `data/E8.5_RNA.h5ad` | T1 训练 |
| `data/E9.5_RNA.h5ad` | T1 训练 |

验证目标 E10.5、测试目标 E12.5 的答案不会随训练数据发放。本地只能用「由 E8.5 预测 E9.5」做代理评分。

---

## 3. 运行步骤

### 步骤 A：审计数据

检查细胞数、基因顺序、类型标签，写出 `outputs/t1/audit.json` 和基因顺序：

```bash
python scripts/00_audit.py
```

看到 E8.5 约 16,787 细胞、E9.5 约 17,057 细胞、32,285 基因、无 `var_names` 冲突即可继续。

### 步骤 B：生成基线预测

```bash
python scripts/01_baselines.py
```

输出在 `outputs/t1/preds/`：

| 文件 | 含义 | 用途 |
|------|------|------|
| `copy_last_E95.h5ad` | 子采样 E8.5 | 本地一步地板（公平） |
| `resample_noise_E95.h5ad` | E8.5 分层重采样 + 噪声 | 本地一步弱基线 |
| `shift_on_E85_to_E95.h5ad` | E8.5 + (mean_E9.5 − mean_E8.5) | **泄漏了 E9.5 均值**，只做诊断 |
| `copy_E95.h5ad` | 子采样 E9.5 | 外推时的 copy_last |
| `E95_plus_delta.h5ad` | E9.5 + 1×Δ | **P2 建议第一个线上提交** |

### 步骤 C：本地评分（veckit）

这不是官网验证分，只检查格式，并用「预测 E9.5、参照 E8.5」比较模型：

```bash
python scripts/04_score_local.py --pred outputs/t1/preds/copy_last_E95.h5ad
```

等价 CLI：

```bash
veckit --task T1 \
  --input outputs/t1/preds/copy_last_E95.h5ad \
  --target data/E9.5_RNA.h5ad \
  --reference data/E8.5_RNA.h5ad
```

`--reference` 必须是 E8.5。省略的话 DE 指标会变成 0，没有意义。

---

## 4. 训练步骤

训练会拟合工作簇组成、64 维 IncrementalPCA，以及 E8.5→E9.5 的残差 OT-CFM。默认 200 epoch，单卡或 CPU 数小时量级。

```bash
python scripts/02_train_flow.py
```

常用参数：

```bash
# 冒烟：只跑 5 个 epoch
python scripts/02_train_flow.py --epochs 5

# 第二周起给 MMD 正则（默认 0，与技术方案一致）
python scripts/02_train_flow.py --epochs 200 --mmd-weight 0.1

# 指定设备
python scripts/02_train_flow.py --device cuda
python scripts/02_train_flow.py --device cpu
```

超参数在 [configs/t1.yaml](configs/t1.yaml)（学习率、隐层、`n_submit`、shift 的 α 等）。改 yaml 后重新训练即可。

训练产物：

| 路径 | 内容 |
|------|------|
| `outputs/t1/composition.json` | 工作簇比例模型 |
| `outputs/t1/pca.joblib` | 表达 PCA |
| `outputs/t1/delta.npy` | mean(E9.5) − mean(E8.5) |
| `outputs/t1/z_85.npy` / `z_95.npy` | 两阶段 PCA 坐标 |
| `outputs/t1/ckpt/flow.pt` | CFM 权重 |

没有这些文件时，`03_predict.py --use-flow` 会报错。

---

## 5. 生成提交 / 本地代理预测

先完成步骤 A、B；带 `--use-flow` 时还要完成步骤 4。

```bash
# 组成外推 + 残差 shift，无 flow（P2 第二份候选）
python scripts/03_predict.py --target 10.5

# 带 flow 的 E10.5（仅当本地一步 MMD 明显好于 E95_plus_delta 再考虑上传）
python scripts/03_predict.py --target 10.5 --use-flow

# 本地一步：从 E8.5 生成假 E9.5，用来和基线比 MMD
python scripts/03_predict.py --target 9.5 --source 8.5
python scripts/03_predict.py --target 9.5 --source 8.5 --use-flow
python scripts/04_score_local.py --pred outputs/t1/preds/pred_E9.5_flow_from8.5.h5ad

# 最终测试阶段（α 默认 3，P3 再用验证集改 configs/t1.yaml 里的 alpha_125）
python scripts/03_predict.py --target 12.5
```

可选参数：`--n 3000`（验证榜允许 1000–5118）、`--out path.h5ad`、`--seed 0`。

默认输出名形如 `outputs/t1/preds/pred_E10.5_comp_from9.5.h5ad`（`comp` 或 `flow`）。

---

## 6. 建议的提交顺序

1. 上传 `outputs/t1/preds/E95_plus_delta.h5ad`，确认格式被收，记下四项指标。
2. 再传 `03_predict.py --target 10.5`（无 flow）的结果。
3. 仅当步骤 C 里带 flow 的 E9.5 代理 **MMD 明显好于** `E95_plus_delta` 时，才上传 `--use-flow` 版本。

判定门与技术方案一致：flow 若打不过简单 shift，线上不要交 flow。

P3（约 2026-10-20）验证答案放出后，用 E10.5 真值重训组成和 α，每个任务最多 2 次正式测试提交。

---

## 7. 目录速查

```text
data/                    原始 .h5ad
configs/t1.yaml          超参数
src/t1/                  库代码
scripts/00_audit.py      审计
scripts/01_baselines.py  地板 / shift 基线
scripts/02_train_flow.py 训练 PCA + CFM
scripts/03_predict.py    生成 .h5ad
scripts/04_score_local.py  本地 veckit
outputs/t1/preds/        预测文件（可提交）
```

---

# Task 2

3D MERFISH 时空预测：每个细胞同时有 panel 表达（`.X`）和 `obsm["spatial_3D"]`。胚胎 setting 与心脏 setting **分模型、分开提交、分数永不平均**。不要对绝对 xyz 做 MSE，也不要把 T1 的 32,285 维网络接到这里。

仓库**不包含**已训练的 OT-CFM 权重。不带 `--use-flow` 的基线与 `full` 可以直接跑；要用 flow 必须先完成本节第 4 步。

详细设计见 [doc/t2_technical_plan.md](doc/t2_technical_plan.md)。

---

## 1. 环境

与 Task 1 共用同一虚拟环境即可：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-t2.txt
```

依赖与 T1 基本相同（`anndata`、`numpy`、`scipy`、`scikit-learn`、`torch`、`veckit`、`pyyaml`、`joblib`）。GPU 可选；几何通道是 numpy / scipy，不需要 GPU。

三个官方 panel 已放在 `panels/`，**不要改顺序或换行**：

| 文件 | 榜单 | 基因数 |
|------|------|--------|
| `panels/T2__embryo__val_interp.genes.txt` | 胚胎验证插值 E7.5 | 498 |
| `panels/T2__heart__val_interp.genes.txt` | 心脏验证插值 E8.5 | 500 |
| `panels/T2__heart__val_extrap.genes.txt` | 心脏验证外推 E10.5 | 500 |

超参数在 [configs/t2.yaml](configs/t2.yaml)（`n_submit`、生长 β、shape 模式、flow 开关等）。改 yaml 后重新跑拟合 / 训练 / 预测即可。

---

## 2. 需要的数据

| 文件 | setting | 角色 |
|------|---------|------|
| `data/E6.75.h5ad` | 胚胎 | 训练 |
| `data/E7.25.h5ad` | 胚胎 | 训练；E7.5 的左锚 |
| `data/E8.0.h5ad` | 胚胎 | 训练；E7.5 的右锚 |
| `data/E8.25_late.h5ad` | 心脏 | 训练；E8.5 的左锚（约 225 MB，体积对不上则重下） |
| `data/E8.75.h5ad` | 心脏 | 训练；E8.5 的右锚 |
| `data/E9.5.h5ad` | 心脏 | 训练；E10.5 的锚 |

E7.5 / E8.5 / E10.5 / E12.5 的答案不随训练数据发放。本地只能用代理任务选模型（见第 3 节步骤 D）。

提交必须同时有 `.X` 和 `obsm["spatial_3D"]`：

| 字段 | 要求 |
|------|------|
| `.X` | `float32`，log 归一化、有限、**非负** |
| `var.index` | 与该榜 panel **逐元素同序**（胚胎 498，心脏 500） |
| `obsm["spatial_3D"]` | `(n, 3)`；坐标系任意 |
| `n_obs` | 推荐 **3000**（胚胎验证 583–5000；心脏插值 1000–17616；心脏外推 1000–25179） |

---

## 3. 运行步骤（无需神经网络）

按 A → B → C → D 顺序。OT-CFM 不是这一段的前置条件。

### 步骤 A：审计数据

检查细胞数、基因顺序、RMS、工作簇映射，写出 `outputs/t2/audit.json`：

```bash
python scripts/10_audit_t2.py
```

应看到约：E6.75 7093×498、E7.25 13295×498、E8.0 31671×500、E8.25 58716×500、E8.75 24826×500、E9.5 53742×500。胚胎 panel 为 498（E8.0 多出的 `Casp4` / `Pnliprp1` 不进胚胎提交）。出现 `missing` / `n_vars` / `Unmapped` 警告时先修数据或簇映射再往下。

### 步骤 B：拟合几何与组成

按 setting 分别拟合 RMS / 主轴、工作簇比例 π(t)、panel 残差 Δ：

```bash
python scripts/12_fit_geometry.py
```

| 路径 | 内容 |
|------|------|
| `outputs/t2/embryo/geometry.json` | 各阶段 RMS、主轴、目标 `r(t*)` |
| `outputs/t2/embryo/composition.json` | 胚胎工作簇组成 |
| `outputs/t2/embryo/delta_725_to_80.npy` | mean(E8.0) − mean(E7.25) |
| `outputs/t2/embryo/delta_675_to_80.npy` | 本地代理用 Δ |
| `outputs/t2/heart/geometry.json` | 同上（心脏） |
| `outputs/t2/heart/composition.json` | 心脏工作簇组成 |
| `outputs/t2/heart/delta_825_to_875.npy` | 插值 Δ |
| `outputs/t2/heart/delta_875_to_95.npy` | 外推 Δ |

没有 `composition.json` 时，`--method full` 会在推理时当场拟合，但建议先跑本步。

### 步骤 C：生成基线预测

```bash
python scripts/11_baselines_t2.py
```

全部 n=3000、panel 已对齐、同时写 `.X` 和 `spatial_3D`。

**官方三榜**

| 文件 | 榜单 | 方法 |
|------|------|------|
| `outputs/t2/embryo/copy_last_E75.h5ad` | 胚胎 E7.5 | 分层拷贝 E7.25 |
| `outputs/t2/embryo/scale_copy_E75.h5ad` | 胚胎 E7.5 | 拷贝后缩放到目标 RMS（**放大**） |
| `outputs/t2/embryo/shift_scale_E75.h5ad` | 胚胎 E7.5 | 残差 shift + 缩放 |
| `outputs/t2/heart/copy_last_E85.h5ad` | 心脏 E8.5 | 拷贝 E8.25 |
| `outputs/t2/heart/scale_copy_E85.h5ad` | 心脏 E8.5 | 拷贝后**缩小**（取样视野收窄，不是器官长大） |
| `outputs/t2/heart/shift_scale_E85.h5ad` | 心脏 E8.5 | shift + 缩小 |
| `outputs/t2/heart/copy_last_E105.h5ad` | 心脏 E10.5 | 拷贝 E9.5 |
| `outputs/t2/heart/scale_copy_E105.h5ad` | 心脏 E10.5 | 第一版 β=0，尺度等于 E9.5 |
| `outputs/t2/heart/shift_scale_E105.h5ad` | 心脏 E10.5 | shift + 缩放 |

**本地代理**（给步骤 D 用，不要当线上文件）

| 文件 | 代理 |
|------|------|
| `outputs/t2/embryo/*_proxy_E725.h5ad` | 用 E6.75+E8.0 假装打 E7.25 |
| `outputs/t2/heart/*_proxy_E95.h5ad` | 用 E8.75 假装打 E9.5 |
| `outputs/t2/heart/*_growth_E875.h5ad` | E8.25→E8.75 生长诊断，只看 TSR 符号 |

### 步骤 D：本地代理评分（veckit）

这不是官网验证分。`--reference` 必须是目标的前一可见阶段，否则 DE 项会假成 0。

```bash
python scripts/14_score_local_t2.py --proxy embryo_interp
python scripts/14_score_local_t2.py --proxy heart_growth
python scripts/14_score_local_t2.py --proxy heart_extrap
python scripts/14_score_local_t2.py --proxy heart_interp_gate
```

| `--proxy` | 在做什么 | 注意 |
|-----------|----------|------|
| `embryo_interp` | 假 E7.25 vs 真 E7.25，ref=E6.75 | 有答案；默认评 `scale_copy_proxy_E725.h5ad` |
| `heart_extrap` | 假 E9.5 vs 真 E9.5，ref=E8.75 | 标签几乎重标，MMD/NFS 会难看 |
| `heart_growth` | E8.25→E8.75 | **只看几何是否缩小**，不要当成心脏萎缩 |
| `heart_interp_gate` | 检查 E8.5 预测的 RMS | 本地无 E8.5 真值；RMS 应落在 217–354，log 线性约 277 |

指定自己的预测，并做 NFS 打乱诊断（同一份 `.X`，xyz 置换后 `neighborhood_mmd` 应变差）：

```bash
python scripts/14_score_local_t2.py --proxy embryo_interp \
  --pred outputs/t2/embryo/pred_E7.25_full_isotropic.h5ad --shuffle
```

等价 CLI 示例：

```bash
veckit --task T2 --setting embryo \
  --input outputs/t2/embryo/scale_copy_proxy_E725.h5ad \
  --target data/E7.25.h5ad --reference data/E6.75.h5ad

veckit --task T2 --setting heart \
  --input outputs/t2/heart/scale_copy_proxy_E95.h5ad \
  --target data/E9.5.h5ad --reference data/E8.75.h5ad
```

---

## 4. 训练步骤（可选 OT-CFM）

每个 setting **单独**在 MERFISH panel 上训练残差 OT-CFM，**不复用** T1 的 32k 维 `flow.pt`。本步会拟合 IncrementalPCA-32，并在相邻 hop 的共享工作簇内做 OT 配对：

- 胚胎：E6.75→E7.25、E7.25→E8.0
- 心脏：E8.25→E8.75、E8.75→E9.5

默认 200 epoch；32 维上单卡或 CPU 通常远短于 T1。仓库里**没有**现成 `pca.joblib` / `flow.pt`，必须自己跑。不跑本步也可以提交 `copy_last` / `scale_copy` / `shift_scale` / 无 flow 的 `full`。

```bash
# 两个 setting 都训（默认）
python scripts/15_train_flow_t2.py

# 只训一个 setting
python scripts/15_train_flow_t2.py --setting embryo
python scripts/15_train_flow_t2.py --setting heart

# 冒烟：只跑 5 个 epoch
python scripts/15_train_flow_t2.py --setting embryo --epochs 5

# 第二周起可给 MMD 正则（默认 0，与技术方案一致）
python scripts/15_train_flow_t2.py --epochs 200 --mmd-weight 0.1

# 指定设备
python scripts/15_train_flow_t2.py --device cuda
python scripts/15_train_flow_t2.py --device cpu
```

隐层、学习率、`z_noise`、`euler_steps` 等在 `configs/t2.yaml` 的 `flow:` 段。`flow.use` 默认 `false`（即使训完也不会自动启用）。

训练产物（每个 setting 一份）：

| 路径 | 内容 |
|------|------|
| `outputs/t2/{setting}/pca.joblib` | panel PCA-32 |
| `outputs/t2/{setting}/ckpt/flow.pt` | 该 setting 的 CFM 权重 |

没有这两文件时，`13_predict_t2.py --use-flow` 会报错。判定门：本地代理 MMD **不明显好于** 无 flow 的 `full` 时，不要提交带 flow 的版本。2026-09-03 评估见 [doc/t2_flow_eval.md](doc/t2_flow_eval.md)（结论：先不要默认交 `--use-flow`）。

---

## 5. 生成提交 / 本地代理预测

先完成步骤 A、B。`--method full` 建议已有步骤 B 的 `composition.json`。带 `--use-flow` 时还要完成第 4 步。

```bash
# 官方三榜：几何缩放基线
python scripts/13_predict_t2.py --setting embryo --target 7.5 --method scale_copy
python scripts/13_predict_t2.py --setting heart --target 8.5 --method scale_copy
python scripts/13_predict_t2.py --setting heart --target 10.5 --method scale_copy

# 残差 shift + 缩放
python scripts/13_predict_t2.py --setting embryo --target 7.5 --method shift_scale
python scripts/13_predict_t2.py --setting heart --target 8.5 --method shift_scale
python scripts/13_predict_t2.py --setting heart --target 10.5 --method shift_scale

# 组成 + 按簇放点（无 flow）
python scripts/13_predict_t2.py --setting embryo --target 7.5 --method full
python scripts/13_predict_t2.py --setting heart --target 8.5 --method full
python scripts/13_predict_t2.py --setting heart --target 10.5 --method full

# 带 OT-CFM（仅 full；必须先跑 scripts/15_train_flow_t2.py）
python scripts/13_predict_t2.py --setting embryo --target 7.5 --method full --use-flow
python scripts/13_predict_t2.py --setting heart --target 8.5 --method full --use-flow
python scripts/13_predict_t2.py --setting heart --target 10.5 --method full --use-flow

# 本地代理（用来和步骤 D 对分，不是线上文件）
python scripts/13_predict_t2.py --setting embryo --target 7.25 --method full
python scripts/13_predict_t2.py --setting heart --target 9.5 --method full
python scripts/13_predict_t2.py --setting heart --target 8.75 --method scale_copy

# 隐藏测试阶段（P3 后；E12.5 的 β 在验证答案放出前不要猜）
python scripts/13_predict_t2.py --setting embryo --target 7.75 --method full
python scripts/13_predict_t2.py --setting heart --target 12.5 --method full
```

`--method`：

| 值 | 表达 | 几何 |
|----|------|------|
| `copy_last` | 分层拷贝最近可见阶段 | 原样 xyz |
| `scale_copy` | 同上 | 去质心后缩放到预测 RMS |
| `shift_scale` | `clip(src + αΔ, 0)` | 同 `scale_copy` |
| `full` | 工作簇组成抽样 + 残差 Δ；可选 OT-CFM | 缩放后的同簇点云 + 抖动 |

仅 `full` 可用的开关：

| 参数 | 含义 |
|------|------|
| `--shape isotropic\|anisotropic\|tps` | 默认 **anisotropic**；`allow_tps: false` 时 `--shape tps` 会被拒绝 |
| `--mix-anchors` | 插值时按时间权重混合两侧点云（会叠两个胚胎，默认关） |
| `--use-flow` | panel PCA 上类型内 OT-CFM，再加残差 Δ；P2 不要交 |
| `--jitter` | 覆盖 `jitter_frac`（消融用；默认 0.15 已够） |

其它参数：`--n 3000`、`--out path.h5ad`、`--seed 0`。

默认输出名：`outputs/t2/{setting}/pred_E{t}_{method}.h5ad`。`full` 会带上 shape，例如 `pred_E7.5_full_isotropic.h5ad`；加 flow 则为 `pred_E7.5_full_isotropic_flow.h5ad`。

生成后可用步骤 D 对代理目标打分：

```bash
python scripts/14_score_local_t2.py --proxy embryo_interp \
  --pred outputs/t2/embryo/pred_E7.25_full_isotropic.h5ad --shuffle
python scripts/14_score_local_t2.py --proxy heart_extrap \
  --pred outputs/t2/heart/pred_E9.5_full_isotropic.h5ad
python scripts/14_score_local_t2.py --proxy heart_interp_gate \
  --pred outputs/t2/heart/pred_E8.5_full_isotropic.h5ad
```

---

## 6. 建议的提交顺序

三个榜**分别**提交，每次只改一件事，不要把三榜分数平均来选「一个总模型」。P2 **停 TPS、停 `--use-flow`**；心脏外推 β 等线上 `scale_log_ratio` 再调。

若前面的地板还没传：

1. `copy_last`：`outputs/t2/embryo/copy_last_E75.h5ad`、`heart/copy_last_E85.h5ad`、`heart/copy_last_E105.h5ad`
2. `scale_copy`：应赢 TSR。胚胎放大、心脏插值缩小、心脏外推 β=0。
3. `shift_scale`：本地 DES/DCS 优于 copy，TSR 不回退。
4. `full` isotropic（成对 `(X,xyz)`）：NFS 门已过。
5. **下一份（当前建议）：** 无 flow 的 `full` anisotropic，已放在 `outputs/t2/submit/`：

| 榜单 | 上传这个文件 |
|------|----------------|
| `T2:embryo:val_interp` | `outputs/t2/submit/T2_embryo_val_interp.h5ad`（498 基因，RMS≈192） |
| `T2:heart:val_interp` | `outputs/t2/submit/T2_heart_val_interp.h5ad`（500 基因，RMS≈277） |
| `T2:heart:val_extrap` | `outputs/t2/submit/T2_heart_val_extrap.h5ad`（500 基因，RMS≈335，β=0） |

清单见 `outputs/t2/submit/manifest.json`。拿到官方 TSR 后，只改 `configs/t2.yaml` 的 `heart_extrap_beta_105` 再生成外推，不要同时改形状或 flow。

P3（约 2026-10-20）验证答案放出后，用真值重训组成、β、α；每个 setting 的测试只留 2 个版本（稳健：scale+shift+按簇；激进：+TPS/flow 仅当验证集门控通过）。

---

## 7. 目录速查

```text
data/                         原始 .h5ad
configs/t2.yaml               T2 超参数
panels/T2__*.genes.txt        三个榜单 panel
src/t2/                       库代码（与 T1 独立，含自己的 flow / pca）
scripts/10_audit_t2.py        审计
scripts/12_fit_geometry.py    几何 + 组成 + Δ
scripts/11_baselines_t2.py    copy / scale / shift 基线
scripts/15_train_flow_t2.py   panel PCA + OT-CFM（可选）
scripts/13_predict_t2.py      生成 .h5ad
scripts/14_score_local_t2.py  本地 veckit / RMS 门控
outputs/t2/audit.json
outputs/t2/embryo/            胚胎拟合产物与预测
outputs/t2/heart/             心脏拟合产物与预测
```
