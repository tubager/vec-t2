# Task 2 技术方案（可执行）

- 对应总计划：[model_training_plan.md](model_training_plan.md)
- OT-CFM 本地评估（2026-09-03）：[t2_flow_eval.md](t2_flow_eval.md)
- P2 验证榜提交（2026-09-05）：[t2_p2_val_2026-09-05.md](t2_p2_val_2026-09-05.md)
- P2 冲前十计划（2026-09-07）：[t2_p2_plan_2026-09-07.md](t2_p2_plan_2026-09-07.md)
- 官方任务页：[评分 Task 2](https://virtualembryo.ai/challenge/evaluation?section=scoring&task=2) · [数据](https://virtualembryo.ai/challenge/data) · [panel 索引](https://virtualembryo.ai/challenge/panels/index.json)
- 本地评分：`pip install veckit`（当前 0.1.2）
- 文档日期：2026-09-02
- 术语：文末 [术语表](#术语表)；总计划里还有通用缩写。T1 表达通道细节见 [t1_technical_plan.md](t1_technical_plan.md)。

本文只覆盖 **T2：3D MERFISH 的时空预测（表达 + `spatial_3D`）**。两个 setting（全胚胎 / 心脏）**分开建模、分开提交、分数永不平均**。目标是：按下面的目录、命令和判定门，能从零跑通「地板基线 → 本地代理评分 → 生成三个验证榜的提交文件」。

---

## 1. 任务定义（用本地文件钉死）

给定已见阶段，生成**目标阶段的细胞群体**：每个细胞同时带 panel 表达和三维坐标。不是平均细胞，不是配准到某一固定图谱坐标系，也不是预测细胞数。

两个 setting 覆盖的组织不同：原肠胚窗口可以切完整胚胎；心脏窗口胚胎已经太大，剖面集中在心脏及相关周边。**不要把胚胎点云和心脏点云放进同一个坐标系或同一个生长模型。**

### 1.1 胚胎 setting（纯插值）

覆盖原肠胚形成。验证和测试都落在训练区间内部，**没有外推轴**。组成周转最快，是更难的一边。决赛的插值技能看隐藏的 E7.75，不看心脏 E8.5。

| 阶段 | 角色 | 本地文件 | 实测 |
|------|------|----------|------|
| E6.75 | 训练 | `data/E6.75.h5ad` | 7,093 × **498**，18 类，RMS 122.7 |
| E7.25 | 训练 | `data/E7.25.h5ad` | 13,295 × **498**，18 类（与 E6.75 **同名全集**），RMS 147.3 |
| E7.5 | 验证插值（P2/P3 前只给分） | 无 | 落在 E7.25–E8.0 之间 |
| E7.75 | 隐藏测试插值 | 无 | 同样夹在 E7.25–E8.0 |
| E8.0 | 训练（右锚点） | `data/E8.0.h5ad` | 31,671 × **500**，26 类，RMS 326.6 |

E6.75 与 E7.25 的 `var_names` **完全同名同序**。E8.0 在保持这 498 个基因相对顺序的前提下，多了 `Casp4`（列 67）和 `Pnliprp1`（列 361）。验证榜 panel 是三者的有序交集，因此是 **498 基因**，即使 E7.5 自己测了那两个基因也不能交 500——参照阶段从未观测过的基因无法打 DE。

E7.25 相对 E6.75：类型名 100% 共享。E8.0 相对 E7.25：同名共享 11 类；按细胞数，E8.0 里只有 **51.9%** 的细胞其类型名在 E7.25 出现过。其余是新标签（FHF/SHF/JCF、前脑、前肠分区等），不要把「标签集合差」当成生物学周转的全部。

### 1.2 心脏 setting（插值与外推分开计分）

| 阶段 | 角色 | 本地文件 | 实测 |
|------|------|----------|------|
| E8.25 | 训练 | `data/E8.25_late.h5ad` | 58,716 × 500，33 类，RMS 354.1；文件 225.1 MB，与数据页 225 MB 一致（完整） |
| E8.5 | 验证插值 | 无 | 落在 E8.25–E8.75 之间 |
| E8.75 | 训练；也是 T3 的 WT 参照 | `data/E8.75.h5ad` | 24,826 × 500，33 类（与 E8.25 **同名全集**），RMS **216.9** |
| E9.5 | 训练 | `data/E9.5.h5ad` | 53,742 × 500，22 类，RMS 335.0 |
| E10.5 | 验证外推 | 无 | 提交后打分 |
| E12.5 | 隐藏测试外推 | 无 | P3 起最多 2 次正式提交 |

三个心脏训练文件的 `var_names` **完全同名同序**，且与胚胎 E8.0 的 500 基因同序。心脏两个验证榜的 panel sha256 相同，都是这 500 个。

E8.25 与 E8.75：33 个类型名完全一致，细胞级共享 100%。E9.5 只与它们同名共享 **5** 类（`NCC`, `Peri`, `V-CM`, `aPHM`, `pPHM`）；按细胞数，E9.5 里只有 **32.3%** 的细胞其类型名在 E8.75 出现过。这和 T1 一样是词汇未对齐，不是「心脏细胞换了 70%」。

**几何上最重要的实测：** E8.25 的 RMS（354）**大于** E8.75（217）。不是心脏在缩小，是取样视野在收窄——E8.25 仍有 13.7% `EXE-Endoderm` 和大量胚外/尿囊，E8.75 变成以神经管 + 心肌为主。心脏插值的 `copy_last`（拷 E8.25）官方地板 `scale_log_ratio = +0.3284`，就是「预测偏大」。插值必须**缩小**，不能按「发育就会变大」去放大。

心脏外推的 `copy_last`（拷 E9.5）地板 `scale_log_ratio = −0.268`，预测偏小，E10.5 确实更大。但 **不要**把 E8.75→E9.5 的 log 斜率（约 0.58 / 天）直接乘到 E10.5：那样会得到 RMS≈598，而地板反推的目标大约 438。E8.25→E9.5 的斜率甚至为负（FOV 伪影）。外推生长只准用 **E9.5 为锚、用验证榜的 TSR 调一步步长**，不要用跨越 FOV 变化的平均斜率。

### 1.3 六个文件共用的矩阵性质

`.X` 为 float32 CSC、`uns["log1p"]` 已做、抽样非负（min=0）、无 NaN；3000 细胞抽样最大值约 8.3–9.2，零值约 87–97%。`obs` 至少有 `celltype`；E8.0 及心脏文件另有 `cm_celltype`，但绝大多数细胞标成 `Unknown`，**工作簇用 `celltype`，不用 `cm_celltype`。**

`obsm` 同时有 `spatial_2D` 和 `spatial_3D`。提交只要 `spatial_3D`（`[n, 3]`，本地为 float64、已近似去质心）。`spatial_2D` 只供可视化。坐标是**每个胚胎局部坐标系**，阶段之间没有配准。

### 1.4 提交契约

一份 AnnData `.h5ad`，**表达和坐标缺一不可**（缺 `spatial_3D` 在打分前拒收）：

| 字段 | 要求 |
|------|------|
| `.X` | `float32`，`[n, G]`，log 归一化、有限、**非负**；稀疏或稠密均可。`G` 由**该榜单** panel 决定，不是训练文件 |
| `var.index` | 与榜单 panel **逐元素同序**。胚胎验证是 498，心脏两个验证榜是 500 |
| `obsm["spatial_3D"]` | `[n, 3]`；坐标系任意（评分对平移/旋转不变；镜像是声明的盲区） |
| `obs["celltype"]` | 可有，**评分忽略**。官方用冻结分类器重新打类型 |
| `n_obs` | 见下表。推荐 **3,000**（MMD 最多抽 2,000；细胞数不是预测目标，也不是组织大小） |

| 榜单 | 基因 | 细胞范围 | 推荐 | panel 文件 |
|------|------|----------|------|------------|
| `T2:embryo:val_interp`（E7.5） | 498 | 583–5,000 | 3,000 | [T2__embryo__val_interp.genes.txt](https://virtualembryo.ai/challenge/panels/T2__embryo__val_interp.genes.txt) |
| `T2:heart:val_interp`（E8.5） | 500 | 1,000–17,616 | 3,000 | [T2__heart__val_interp.genes.txt](https://virtualembryo.ai/challenge/panels/T2__heart__val_interp.genes.txt) |
| `T2:heart:val_extrap`（E10.5） | 500 | 1,000–25,179 | 3,000 | [T2__heart__val_extrap.genes.txt](https://virtualembryo.ai/challenge/panels/T2__heart__val_extrap.genes.txt) |

禁止：只交 `.X`；交胚胎 500 基因；把平均细胞复制 n 次；对绝对 xyz 做 MSE 去「对齐」某个阶段；把细胞数当成胚胎大小；混用胚胎/心脏坐标。

### 1.5 评分（必须对这四组优化）

参照阶段：目标的**前一可见阶段**（胚胎 E7.5 的 ref 是 E7.25；心脏插值 E8.5 的 ref 是 E8.25；心脏外推 E10.5 的 ref 是 E9.5）。本地代理见第 4 节。

| 题组 | 权重 | 指标 | 直觉 |
|------|------|------|------|
| 表达变化 | 25% | DES↑ 0.50 + DCS↑ 0.50 | 哪些基因动了、方向对不对（相对参照） |
| 细胞状态 | 25% | MMD↓ 0.60 + CSS↓ 0.40 | 类型与比例、基因共变 |
| 形状与尺度 | 25% | SDD↓ / ODS↑ / TSR→0 各 1/3 | 旋转不变的形态 + 组织是不是差不多大 |
| 局部空间组织 | 25% | NFS↓（kNN-15 邻域 pseudobulk 的 MMD） | 细胞旁边该是什么邻居；**这是表达和坐标必须联合对的地方** |

排行榜 skill：地板 `copy_last` = 50 分。**高于 50 才算学到变化。** 官方 `spatiotemporal_ode` 的分几乎全来自**解析生长率**，不是 ODE 本身——几何通道先做对生长，再考虑形状网络。

验证榜锚点（原始单位，来自 `index.json`）：

**胚胎 val_interp**

| 指标 | 地板 | 天花板 |
|------|------|--------|
| `de_score` | 0 | 0.8413 |
| `de_direction` | 0 | 0.9185 |
| `mmd_u` | 0.08571 | 0.00307 |
| `variogram` | 0.053533 | 0.001264 |
| `d2_shape` | 0.05306 | 0.00268 |
| `occupancy_dice` | 0.7047 | 0.7661 |
| `scale_log_ratio` | −0.3063 | 0.0053 |
| `neighborhood_mmd` | 0.21105 | 0.01128 |

**心脏 val_interp**

| 指标 | 地板 | 天花板 |
|------|------|--------|
| `de_score` | 0 | 0.8182 |
| `de_direction` | 0 | 0.9553 |
| `mmd_u` | 0.0207 | 0.00087 |
| `variogram` | 0.023828 | 0.001021 |
| `d2_shape` | 0.03079 | 0.00412 |
| `occupancy_dice` | 0.6748 | 0.8796 |
| `scale_log_ratio` | **+0.3284** | −0.0119 |
| `neighborhood_mmd` | 0.05728 | 0.00432 |

**心脏 val_extrap**

| 指标 | 地板 | 天花板 |
|------|------|--------|
| `de_score` | 0 | 0.942 |
| `de_direction` | 0 | 0.9915 |
| `mmd_u` | 0.02455 | 0.00011 |
| `variogram` | 0.031712 | 0.000696 |
| `d2_shape` | 0.00933 | 0.00342 |
| `occupancy_dice` | 0.7747 | 0.9465 |
| `scale_log_ratio` | −0.268 | 0.0074 |
| `neighborhood_mmd` | 0.07184 | 0.0004 |

形状项会先把点云去质心、PCA 对齐到主轴、再除以自身 RMS，只留形状；TSR 单独看 RMS 对数比。PCA 轴符号在 `det = +1` 的四个翻转上取最优，**不奖励镜像，也检测不了镜像**（laterality 盲区）。不要为左右不对称去优化。

### 1.6 外部数据与保护窗

T2 **不**鼓励像 T1 那样用外部 scRNA 补空间。不可用的实测（含图谱里同窗的空间数据）：

- 胚胎：E7.5–E7.75
- 心脏：E8.25–E8.75（插值窗），以及 E10.5、E12.5

近邻规则：外部数据若比最近已发布训练阶段更靠近 held-out 阶段，视为作弊。

---

## 2. 方法总览（三块，缺一不可）

```text
胚胎训练 或 心脏训练（分模型，永不混坐标系）
    │
    ├─ A. 表达通道（50%）  组成/出生 + 残差 shift + 可选 panel PCA flow
    ├─ B. 几何通道（25%）  Procrustes 规范化 + RMS/各向异性生长 + 旋转不变形状
    └─ C. 联合通道（25%）  按工作簇把细胞放到缩放后的同簇点云上（NFS）
    │
    ▼
~3,000 × G 的 .X  +  spatial_3D  →  三个榜单各一份 .h5ad
```

判定门（实现时遵守，过不了就停在更简单的件上）：

- **几何**：若「各向同性 RMS 缩放的 `copy_last`」在本地代理上 TSR/SDD 不优于未缩放的 `copy_last`，先别上 TPS / 形状 flow。
- **表达**：若 panel 上的 flow 本地 MMD 不优于 `pseudobulk_shift`，提交只用 shift。
- **NFS**：若「按簇拷贝缩放后的点云」不优于把坐标随机打乱，再加邻域图；打乱必须明显变差，否则联合通道没接上。

不采用：端到端微调 scGPT 当提交模型；对绝对 xyz 做 MSE；把 T1 的 32,285 维接到 T2；把 T1 心脏解剖组成当胚胎先验。

---

## 3. 仓库布局（按此创建；本文不实现代码）

在项目根目录；原始数据在 `data/`：

```text
Embryo/
  data/
    E6.75.h5ad  E7.25.h5ad  E8.0.h5ad          # 胚胎
    E8.25_late.h5ad  E8.75.h5ad  E9.5.h5ad     # 心脏
  doc/
    model_training_plan.md
    t1_technical_plan.md
    t2_technical_plan.md                       ← 本文
  configs/t2.yaml
  panels/
    T2__embryo__val_interp.genes.txt
    T2__heart__val_interp.genes.txt
    T2__heart__val_extrap.genes.txt
  src/t2/
    __init__.py
    io.py            读盘、按榜单 panel 重排、写提交（.X + spatial_3D）
    clusters.py      胚胎 / 心脏两套工作簇
    geometry.py      canonicalize、rms_radius、各向异性缩放
    growth.py        log RMS(t)、主轴长度比
    shape.py         点云插值（缩放 → 各向异性 → TPS/OT，按门控升级）
    neighborhood.py  按簇采样 xyz + 抖动
    expression.py    调用 panel PCA / 组成 / shift / 可选 OT-CFM
    flow.py          非自治 OT-CFM（panel PCA，与 T1 分模型）
    pca.py           IncrementalPCA-32
    ot.py            簇内 Hungarian 配对
    infer.py         指定 setting + 目标 t 生成 n 个细胞
  scripts/
    10_audit_t2.py
    11_baselines_t2.py
    12_fit_geometry.py
    13_predict_t2.py
    14_score_local_t2.py
    15_train_flow_t2.py
  outputs/t2/
    embryo/  heart/
```

Python 3.10–3.13。依赖在 T1 之上加 `veckit`（T2 的 `--setting`）。GPU 可选；几何通道是 numpy / scipy，不需要 GPU。

`configs/t2.yaml` 建议初值：

```yaml
n_submit: 3000
pca_dim: 32              # 500 基因不需要 T1 的 64
seed: 0
knn_k: 15                # 与官方 NFS 一致
jitter_frac: 0.15        # 相对中位 kNN 距

paths:
  embryo:
    E6.75: data/E6.75.h5ad
    E7.25: data/E7.25.h5ad
    E8.0: data/E8.0.h5ad
    panel_val_interp: panels/T2__embryo__val_interp.genes.txt
  heart:
    E8.25: data/E8.25_late.h5ad
    E8.75: data/E8.75.h5ad
    E9.5: data/E9.5.h5ad
    panel_val_interp: panels/T2__heart__val_interp.genes.txt
    panel_val_extrap: panels/T2__heart__val_extrap.genes.txt
  preds: outputs/t2

time:
  embryo: {t_675: 6.75, t_725: 7.25, t_75: 7.5, t_775: 7.75, t_80: 8.0}
  heart:  {t_825: 8.25, t_85: 8.5, t_875: 8.75, t_95: 9.5, t_105: 10.5, t_125: 12.5}

growth:
  metric: rms_radius     # 不要用包围盒
  embryo_interp_anchors: [7.25, 8.0]
  heart_interp_anchors: [8.25, 8.75]
  heart_extrap_anchor: 9.5
  heart_extrap_alpha_105: 1.0   # 相对「E9.5→推断 E10.5」的步长，用线上 TSR 调
  heart_extrap_alpha_125: null  # P3 再用验证集选

shift:
  clip_min: 0.0
  # 心脏外推可选：把 T1 Δ 投影到 500 基因，权重从 0 加起
  t1_delta_weight: 0.0
```

---

## 4. 本地评分协议（在没有验证答案时）

`veckit` **不是**真实验证分。它只保证格式和指标代码能跑。必须自己规定代理任务，否则无法选模型。

命令形态（`--reference` 必须是目标的前一可见阶段；省略则 DE 项会假成 0）：

```bash
veckit --task T2 --setting embryo \
  --input pred.h5ad --target data/E7.25.h5ad --reference data/E6.75.h5ad

veckit --task T2 --setting heart \
  --input pred.h5ad --target data/E9.5.h5ad --reference data/E8.75.h5ad
```

### 4.1 胚胎插值代理（有答案）

留出 E7.25：只用 E6.75 + E8.0 插值生成「假 E7.25」，对真实 E7.25 打分，ref = E6.75。时间权重 `w = (7.25−6.75)/(8.0−6.75) = 0.4`。这与真实任务（在 7.25 与 8.0 之间插 E7.5）同构，只是换了一个被夹住的点。

地板对照：把 E6.75 子采样当预测去对 E7.25。`de_score` / `de_direction` 应在 0 附近；`scale_log_ratio` 应为负（E6.75 更小）。

### 4.2 心脏一步外推代理（有答案）

用 E8.75 生成「假 E9.5」，对真实 E9.5 打分，ref = E8.75。这是外推技能的本地版，但 **词汇在 E9.5 几乎重标**，MMD/NFS 会很难看——仍然有用，因为线上 E10.5 也会面对「下一阶段标签集未知」。

另报 E8.25→E8.75 作为**生长诊断**（类型名完全对齐，几何却在缩小）：用来确认 TSR 符号，不要把它当「心脏在萎缩」的生物学结论。

### 4.3 心脏插值：不能本地假装打 E8.5 的生物分

本地没有 E8.5。不要拿 E7.75 RNA 或其它模态来冒充。门控只用几何自洽：

- 预测的 RMS 必须落在 216.9 与 354.1 之间（log 空间插值目标约 **277**；地板反推约 **255**，数量级一致即可）
- 各向同性缩放到该 RMS 后，相对「未缩放的 E8.25 copy」应改善拟 TSR（`log(r_pred/277)` 的绝对值变小）
- 真正的 DES/MMD/NFS 等线上 E8.5

### 4.4 不要做的事

没有第三个心脏时间点在 E9.5 之后，**不能**在本地假装打 E10.5 的生物分。选外推模型看：4.2 的一步分 + 提交后的官方 E10.5 TSR/DES。P3 验证答案放出后再重训。

---

## 5. 第 0 周：审计与地板（必须先完成）

### 5.1 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install anndata numpy scipy pandas scikit-learn torch veckit pyyaml joblib
```

从官网把三个 T2 panel 放到 `panels/`，勿手改顺序或换行。

### 5.2 `scripts/10_audit_t2.py` 必须打印并写入 `outputs/t2/audit.json`

应与第 1 节一致：

- 细胞数 / 基因数：7093×498、13295×498、31671×500、58716×500、24826×500、53742×500
- 胚胎 498 = E8.0 去掉 `Casp4`/`Pnliprp1` 后的同序子集；心脏三文件同序；E8.0 与心脏 500 同序
- 与三个 panel 文件逐行 equal（不等则写提交前 `reindex`）
- `X.min() >= 0`，无 NaN/Inf；`spatial_3D` 为 `(n, 3)`，无 NaN
- 各阶段 RMS、三主轴标准差、类型表、共享/独有集合
- E8.25 文件体积约 225 MB

若 panel 与训练文件不一致：**训练仍用训练矩阵，写提交前按 panel 重排**；胚胎缺的 `Casp4`/`Pnliprp1` 直接不写进胚胎提交。

### 5.3 `scripts/11_baselines_t2.py`

所有输出：`n=3000`、panel 已对齐、同时写 `.X` 和 `obsm["spatial_3D"]`。

**`copy_last(adata, n, seed)`**  
分层子采样，拷贝表达和坐标。胚胎 E7.5 代理用 E7.25；心脏 E8.5 用 E8.25；心脏 E10.5 用 E9.5。

**`scale_copy(adata, n, r_target, seed)`**  
`copy_last` 之后把坐标去质心再乘 `r_target / rms(src)`。这是几何第一件，应单独赢 TSR。

**`shift_scale(src, ref_prev, ref_next, r_target, n, alpha=1)`**  
表达：`clip(src + α · (mean(ref_next)−mean(ref_prev)), 0)`。几何：`scale_copy`。官方 `pseudobulk_shift` **不动坐标**；我们主动改几何。

`r_target` 取值：

| 目标 | 公式 | 本地数值 |
|------|------|----------|
| 胚胎 E7.5 | log 线性插值 RMS(7.25) 与 RMS(8.0) | ≈ 192 |
| 胚胎 E7.75 | 同上，t=7.75 | ≈ 250 |
| 心脏 E8.5 | log 线性插值 RMS(8.25) 与 RMS(8.75) | ≈ 277 |
| 心脏 E10.5 | 先交 `scale_copy` 到「E9.5 的 RMS × exp(β)」；β 初值用线上 TSR 反推，不要用 8.75→9.5 斜率 | 见第 7.2 节 |

---

## 6. 模块 A：表达通道

复用 [t1_technical_plan.md](t1_technical_plan.md) 的组成 / 残差 shift / 可选 OT-CFM，但基因是 panel（498 或 500），PCA 维数 32。**不要**把 32,285 维提交接到这里。心脏外推可以把 T1 学到的 Δ 投影到 panel 作弱先验（`t1_delta_weight` 从 0 加起）；胚胎不要用 T1 的心脏解剖组成。

### 6.1 不要直接外推标签名

标签词汇未对齐。做法与 T1 相同：映射到工作簇 → 在簇上插值/外推 π → 簇内采样表达 → 提交不写 `celltype`。

**插值（胚胎 E7.5、心脏 E8.5）：** 两侧锚点都有，

```text
π(t*) = (1 − w) · π(t_left) + w · π(t_right)
w = (t* − t_left) / (t_right − t_left)
```

在工作簇上做，再归一化。心脏 E8.25/E8.75 类型名已经对齐，工作簇几乎是恒等映射。胚胎 E7.5 靠近 E7.25（w=(7.5−7.25)/(8.0−7.25)=1/3），组成应更像 E7.25，不要对半混 E8.0 的新标签。

**外推（心脏 E10.5 / E12.5）：** 冻结 π(E9.5)，不要对近零比例做无约束 log-slope（T1 已经因此把 π 崩成出生簇各 50%）。出生簇从 E9.5 库采样加噪声；判定为取样 FOV 丢失的簇（E8.25 的大量 EXE/尿囊）在 t≥9.5 保持为 0，不要从 E8.25 线性「长回来」。

### 6.2 胚胎工作簇（第一版）

E6.75 与 E7.25 十八类同名。E8.0 二十六类。

| 工作簇 | E6.75 / E7.25 | E8.0 | 备注 |
|--------|---------------|------|------|
| exe_endo | EXE-Endoderm | EXE-Endoderm | 一直是最大类之一 |
| exem | ExEM-1, ExEM-2 | ExEM-1, ExEM-2, p-EXEM | 胚外中胚层 |
| exe_ect | EXE-Ectoderm | — | E8.0 为 0 |
| epi | Anterior Epiblast, Caudal Epiblast | Caudal Epiblast | 前上胚层在 E8.0 消失 |
| streak | Primitive Streak | Primitive Streak | 原条衰减 |
| gut | Gut Endoderm | V-FG, D-FG | 内胚层分区 |
| se | aSE, pSE | V-SE, pSE | 表面外胚层 |
| phm | PHM/PAM | aPHM, pPHM | |
| som | SOM | SOM | |
| lpm | LPM | LPM | |
| allantois | Allantois | Allantois, Allantois Endothelium | |
| hem | HEM-Endoth, Blood Progenitor | HEM-Endoth, Intra-Endoth, EXE-Endothlium | 内皮/造血 |
| heart | CP | FHF, SHF, JCF | 心区在 E8.0 展开 |
| neural | — | Forebrain, Hindbrain, d-CSE | E8.0 出生 |
| unknown | Unknown | Unknown | 插值保留比例，不要当出生簇放大 |

### 6.3 心脏工作簇（第一版）

E8.25 与 E8.75 三十三类同名。E9.5 二十二类，同名只剩 5 个。

| 工作簇 | E8.25 / E8.75 | E9.5 | 备注 |
|--------|---------------|------|------|
| CM_V | V-CM | V-CM | 连续 |
| CM_IFT | IFT-CM | A-CM, SV-CM | 流入道→心房/静脉窦（工作假设，审计后可用表达核验） |
| CM_OFT | — | OFT-CM | E9.5 新标签 |
| endo | Endo, Intra-Endoth-1, Intra-Endoth-2 | Chamber-Endo, Cush-EndoMT-Endo, BEC, early-VEC, Great Artery Endoth | 内皮更名 |
| peri | Peri | Peri, Proepi | |
| ncc | NCC | NCC | |
| phm | aPHM, pPHM | aPHM, pPHM | |
| pam | PAM-1, PAM-2, PAM-3, PAM-4 | Dorsal-PAM, Lateral-PAM | |
| se | pSE, V-CSE, d-CSE | SE | |
| gut | D-FG, a-FG, Lateral FG, Gut Endoderm, Hindgut | Foregut, Hepatocytes | |
| nt | Neural Tube, Forebrain | — | E8.75 神经管仍 15%；E9.5 无此名。外推衰减，不要从 15% 线性外推 |
| extra | EXE-Endoderm, ExEM-1/2, Allantois, HEM-Endoth, SOM, LPM, Caudal Epiblast, JCF, Unknown | — | **FOV 丢失**，t≥9.5 保持 0 |
| branch | — | Branch Arch | 出生 |
| st | — | ST, DMP | 出生 |

改了映射要记进 `audit.json`。

### 6.4 残差 shift 与可选 flow

插值：`Δ` 用两侧锚点的均值差按 `w` 加权，加到从左锚点（或按 π 混合的库）采的细胞上，`clip ≥ 0`。

外推：`Δ = mean(E9.5) − mean(E8.75)`（500 维），`α(E10.5)=1`，`α(E12.5)` 等 P3 用验证集网格。可选把 T1 的 32285 维 Δ 按基因名投影到 panel，与 MERFISH Δ 做凸组合；权重要用 4.2 的 DCS 门控，投影伤 DCS 就丢掉。

可选 OT-CFM：在 panel IncrementalPCA-32 上、共享工作簇内配对，非自治，残差空间（先减 Δ）。门控同 T1：本地 MMD 不明显好于「经验重采样 + shift」就**停用 flow**。500 基因上的 CFM 容量很小，预期经常输给 shift。

---

## 7. 模块 B：几何通道（禁止 xyz MSE）

官方提示：`spatiotemporal_ode` 的分几乎全来自解析生长率。先把尺度做对，形状用旋转不变描述符，**不要**回归绝对坐标。

### 7.1 规范化（仅建模用）

对每个阶段的坐标矩阵 `C ∈ R^{n×3}`：

```text
c  = C.mean(axis=0)
X  = C − c
r  = sqrt(mean(||X||^2))          # RMS radius，旋转不变
V  = PCA(X) 的 3 个主轴（右奇异向量）
Z  = X @ V                         # 主轴对齐
Z  = Z / r                         # 单位 RMS，形状比较时用
```

提交可以停在「去质心 + 可选主轴对齐 + 缩放到 r_target」的坐标系；评分会再对齐一次。不要为了和某一训练阶段的原始 xyz 重合去加 MSE。

本地实测 RMS 与主轴标准差：

| 阶段 | RMS | 主轴 std |
|------|-----|----------|
| E6.75 | 122.7 | 100.9, 52.2, 46.3 |
| E7.25 | 147.3 | 105.3, 86.7, 55.6 |
| E8.0 | 326.6 | 226.6, 202.1, 120.3 |
| E8.25 | 354.1 | 245.3, 210.4, 144.9 |
| E8.75 | 216.9 | 136.8, 128.3, 108.9 |
| E9.5 | 335.0 | 244.1, 178.3, 144.4 |

E8.75 三轴更接近等长（更「圆」），与心脏 FOV 收窄一致。

### 7.2 TSR：预测 `r(t*)`

胚胎插值、心脏插值：两侧锚点都有，**log 线性插值**（比线性更贴近地板反推）：

```text
log r(t*) = (1−w) log r(t_left) + w log r(t_right)
```

心脏外推：

```text
log r(E10.5) = log r(E9.5) + β
```

`β` 不要用 8.75→9.5（会到 ~598）或 8.25→9.5（负斜率）。第一版 `β` 取 0（即 `scale_copy` 到 E9.5 自身 RMS，与 `copy_last` 的 TSR 相同），用**线上** `scale_log_ratio` 一次性解开 `β = −(线上 TSR) + 修正`；第二版再把 `β` 写成可配置常数。P2 不要猜 E12.5 的 `β`。

### 7.3 形状：按门控升级，不要一步到位

**第一版（先交这个）：** 拷最近锚点的规范化点云 × `r_target`。胚胎 E7.5 的最近左锚是 E7.25；心脏 E8.5 拷 E8.25 再**缩小**到 r≈277；心脏 E10.5 拷 E9.5 再按 β 缩放。SDD 只看相对距离分布，各向同性缩放不改变 SDD；ODS 在单位 RMS 网格上算，缩放也不改 ODS。所以第一版赢的是 **TSR**，形状项与 `copy_last` 相同——这已经是 25% 里的 1/3。

**第二版：** 三主轴长度比也做 log 线性插值，各向异性缩放。对规范化坐标 `Z` 的三列分别乘 `s1, s2, s3`（归一化使 RMS 仍为 `r_target`）。心脏 E8.25→E8.75 主轴比从 1.69:1.45:1 变到 1.26:1.18:1，插值 E8.5 应介于其间。

**第三版：** 两侧规范化点云做 OT / thin-plate spline 插值。只在第一、二版的本地 SDD/ODS 已经优于 copy 之后才上。外推不要指望 TPS「长出 loop」。

损失：SDD 的 EMD（点对距离直方图）、ODS 的 Dice（16³ 网格、`det=+1` 四个翻转）。**不要**写 `||xyz_pred − xyz_true||^2`。

### 7.4 左右不对称

所有形状项对镜像同样给分。不要把算力花在预测心脏 looping 的左右上。

---

## 8. 模块 C：NFS 联合通道

NFS 对每个细胞取空间 15 近邻的表达均值，再对这组邻域 pseudobulk 做与 MMD 相同的无偏多核差异。它不要求细胞一一对应，但要求「这种表达的细胞周围是那种表达」。

生成顺序：

1. 按 π(t*) 决定每个工作簇要多少细胞，并从表达库采样 `.X`（模块 A）。
2. 对该簇，取几何模板：缩放/变形后的**同簇**点云（插值用按 `w` 混合两侧；外推用 E9.5 该簇）。
3. 从模板无放回采样 xyz（不够则有放回），加 `jitter = 0.15 × 中位 kNN 距` 的各向同性噪声。
4. 拼成 `obsm["spatial_3D"]`。

不要从整个胚胎的一个 blob 采样坐标再随机配给细胞——邻域均值会被抹平，NFS 接近把坐标打乱。

诊断（本地代理必做）：同一份 `.X`，把 xyz 随机置换，`neighborhood_mmd` 应明显变差。若几乎不变，说明联合通道没接上。

---

## 9. 生成三个榜单的推理步骤

`scripts/13_predict_t2.py --setting embryo|heart --target 7.5|8.5|10.5 --n 3000`

1. 选 setting，加载对应训练阶段与 panel。
2. 计算 π(t*)（插值混合；外推冻结尾锚 + extra/nt 为 0）。
3. 按簇采样表达，加 αΔ，clip。
4. 计算 r(t*)；规范化模板点云并缩放（再按门控做各向异性 / TPS）。
5. 按簇放置 xyz + 抖动。
6. `var_names` 对齐该榜单 panel，写 `.X` 与 `spatial_3D`。
7. 断言：`X.shape[1]` 正确、`X.min()>=0`、无 NaN、`spatial_3D.shape==(n,3)`、`n` 在该榜范围内。

### 9.1 P2 提交顺序（每个榜单独一份，每次只改一件事）

1. `copy_last`（最近可见阶段子采样）。用来核对格式和官方地板。
2. `scale_copy`（应赢 TSR。心脏插值必须缩小；胚胎插值必须放大）。
3. 表达 `pseudobulk_shift` + 几何 `scale_copy`。
4. 仅当第 4 节门控通过：组成插值/外推 + 各向异性或 TPS + 按簇放点。
5. 仅当表达 flow 的本地 MMD 明显更好：再交带 flow 的版本。

记下提交 id 与八项原始指标。三个榜的分数不要平均来「选一个总模型」——胚胎插值和心脏外推允许用不同超参。

---

## 10. 训练日程与门控

对齐总计划第 4–7 周。

| 时间 | 产出 | 通过条件 |
|------|------|----------|
| 第 0 天 | `audit.json`、三个 panel 一致、`copy_last` 的 veckit 能跑 | 基因顺序冲突为 0；胚胎提交是 498 列 |
| 第 1–2 天 | 三榜各交 `copy_last` 再交 `scale_copy` | 格式被收；心脏插值 TSR 从 +0.33 向 0 靠近；胚胎插值 TSR 从 −0.31 向 0 靠近 |
| 第 3–5 天 | shift + scale_copy | 本地代理 DES/DCS 优于 copy；TSR 不回退 |
| 第 6–10 天 | 工作簇 + 按簇放点 | 本地 NFS 优于「全局缩放点云 + 随机配表达」；打乱 xyz 使 NFS 变差 |
| 第 11–14 天 | 各向异性缩放；可选 TPS | 本地 SDD/ODS 优于各向同性才上线 |
| 可选 | panel PCA-32 CFM | 本地 MMD 优于 shift 才上线，否则停用 |
| 持续到 10-20 | 小步提交 | 每次只改 r_target / 簇合并 / jitter 之一 |
| 10-20 后 | 用验证真值重训组成、β、α | 每个 setting 的测试只留 2 个版本：稳健（scale+shift+按簇）与激进（+TPS/flow） |

算力：几何是 CPU 分钟级。不必等大 GPU。

---

## 11. 关键代码骨架

### 11.1 规范化、RMS、缩放、写提交

```python
import numpy as np
import anndata as ad

def rms_radius(C):
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    return float(np.sqrt((X**2).sum(axis=1).mean()))

def canonicalize(C):
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:3].T
    r = float(np.sqrt((Z**2).sum(axis=1).mean()))
    return Z, r, Vt[:3]

def scale_cloud(C, r_target):
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    r = float(np.sqrt((X**2).sum(axis=1).mean()))
    return (X * (r_target / max(r, 1e-8))).astype(np.float32)

def anisotropic_scale(Z, axis_std_target, r_target):
    # Z: already PCA-aligned, current RMS ~ 1 or r_src
    s = np.asarray(axis_std_target, dtype=np.float64)
    Y = Z / (Z.std(axis=0, ddof=1) + 1e-8) * s
    Y = Y - Y.mean(axis=0)
    r = float(np.sqrt((Y**2).sum(axis=1).mean()))
    return (Y * (r_target / max(r, 1e-8))).astype(np.float32)

def loglin(t0, t1, t, y0, y1):
    w = (t - t0) / (t1 - t0)
    return float(np.exp((1 - w) * np.log(y0) + w * np.log(y1)))

def write_t2(X, xyz, var_names, path):
    X = np.clip(np.asarray(X, dtype=np.float32), 0, None)
    xyz = np.asarray(xyz, dtype=np.float32)
    assert X.shape[0] == xyz.shape[0] and xyz.shape[1] == 3
    adata = ad.AnnData(X)
    adata.var_names = var_names
    adata.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    adata.obsm["spatial_3D"] = xyz
    adata.write_h5ad(path, compression="gzip")
```

### 11.2 按簇放置坐标

```python
def place_by_cluster(expr_clusters, templates, rng, jitter):
    """templates[k] = (n_k, 3) scaled point cloud for cluster k."""
    xyz = np.empty((len(expr_clusters), 3), dtype=np.float32)
    for i, k in enumerate(expr_clusters):
        P = templates[k]
        xyz[i] = P[rng.integers(0, len(P))]
    nn = median_nn(xyz)  # subsample if needed
    xyz = xyz + rng.normal(0, jitter * nn, size=xyz.shape).astype(np.float32)
    return xyz
```

---

## 12. 失败模式（对照官方 pitfalls 与本地实测）

| 现象 | 原因 | 处理 |
|------|------|------|
| 胚胎提交被拒 gene mismatch | 交了 500 基因或顺序用了 E8.0 | 写盘前对齐 `T2__embryo__val_interp.genes.txt`（498） |
| 缺 `spatial_3D` 被拒 | 只写了 `.X` | `write_t2` 必须写 obsm |
| 心脏插值 TSR 仍是大正数 | 把 E8.25 放大了，或没缩 | 插值目标 RMS≈255–277，必须缩小 |
| 心脏外推 TSR 过冲到很大负数的反面 | 用了 8.75→9.5 斜率（预测 RMS≈598） | 只用 E9.5 当锚，β 用线上 TSR 调 |
| DES 好、NFS 差 | 坐标与类型脱节，全局 blob 采样 | 按工作簇放置；做打乱诊断 |
| SDD/ODS 不动、TSR 已好 | 预期内：各向同性缩放不改形状描述符 | 要形状分再上各向异性 / TPS |
| DES 很好、MMD 很差 | 只学会了均值平移 | 加组成采样；禁止交平均细胞 |
| 对 xyz 做 MSE 后形状分仍差 | 评分忽略平移旋转，MSE 优化了错误的东西 | 删掉坐标 MSE |
| 混用胚胎/心脏坐标 | 两个局部系、两个 FOV | 分模型 |
| 本地一步极好、线上更差 | 过拟合 E8.0 / E9.5 标签名 | 用工作簇；外推冻结 π(E9.5) |
| flow 不如 shift | 预期内 | **提交 shift** |
| 镜像胚胎 | 盲区 | 不必优化 laterality |
| 负值被拒 | 解码或 shift 冲出 0 | 一律 clip |
| E8.25 体积对不上 | 下载不完整 | 本地 225.1 MB 已与数据页一致；若重下后变小再查 |

---

## 13. 与总计划和 T1 的关系

本文是 [model_training_plan.md](model_training_plan.md) 里 T2 的展开。表达通道复用 [t1_technical_plan.md](t1_technical_plan.md) 的 PCA / 组成 / 残差 shift / 可选 OT-CFM，但基因是 MERFISH panel（498 或 500），**禁止**把 32,285 维提交接到 T2。几何通道和 NFS 是 T1 没有的；心脏外推可以把 T1 的 Δ 投影到 panel 作弱先验，胚胎不要用 T1 组成。T3 不要用本文的 `copy_last` 思维（要用 WT→KO 的 delta；形状不计入 T3 总分）。

---

## 术语表

只列本文出现、且总计划未展开到「可对照本地文件」的词。通用缩写（MMD、OT-CFM、DES、veckit、held-out、Procrustes 等）见 [model_training_plan.md 术语表](model_training_plan.md#术语表)。

### 数据与提交

| 术语 | 含义 |
|------|------|
| setting | T2 的两个独立榜：`embryo`（全胚胎原肠胚窗）与 `heart`（心脏中心 MERFISH）。分数不平均。 |
| `val_interp` / `val_extrap` | 插值验证 / 外推验证。胚胎只有 interp；心脏两个都有且分开计分。 |
| 498-gene intersection | 胚胎榜 panel：E6.75、E7.25、E8.0（及目标）都测到的基因有序交集。缺 `Casp4`、`Pnliprp1`。 |
| `E8.25_late.h5ad` | 心脏 setting 的 E8.25 训练文件。本地 58,716 细胞、225.1 MB。 |
| `spatial_3D` | 每细胞 xyz。局部坐标系，未跨阶段配准。提交必带。 |
| `spatial_2D` | 切片平面坐标，仅可视化，不要提交。 |
| `cm_celltype` | 心脏相关细分类；绝大多数为 `Unknown`。工作簇用 `celltype`。 |
| FOV / 取样视野 | 切片覆盖范围。心脏 E8.25→E8.75 RMS 变小，主要是视野收窄而不是器官萎缩。 |
| panel | 该榜单要求的基因符号列表，一行一个、顺序固定。 |

### 几何与评分

| 术语 | 含义 |
|------|------|
| RMS radius | 细胞到质心距离的均方根。TSR 用它当组织尺度；旋转不变。 |
| SDD | Shape Distribution Distance。点对距离（除以自身 RMS）的经验分布之间的 EMD。 |
| ODS | Occupancy Dice Score。规范化后 16³ 网格占据的 Dice，在 `det=+1` 的四个轴翻转上取最大。 |
| TSR | Tissue Scale Ratio。`log(r_pred / r_true)`，0 为尺度正确。 |
| NFS | Neighborhood Fidelity Score。每细胞 15 个空间近邻的表达均值，再对这组向量做无偏 MMD。 |
| laterality | 左右不对称。形状项对镜像同样给分，是声明的盲区。 |
| `scale_copy` | 子采样最近阶段后把点云缩放到预测 RMS。几何第一基线。 |
| β | 心脏外推时加在 `log r(E9.5)` 上的步长。不用跨 FOV 的斜率估计。 |

### 细胞类型缩写（来自 `obs["celltype"]`）

| 标签 | 含义（工作理解） |
|------|------------------|
| EXE-Endoderm / EXE-Ectoderm / ExEM | 胚外内胚层 / 外胚层 / 中胚层。胚胎 setting 占比高；心脏后期 FOV 里几乎消失。 |
| Anterior / Caudal Epiblast | 前 / 尾上胚层。原肠胚早期。 |
| Primitive Streak | 原条。 |
| PHM/PAM, aPHM, pPHM | 咽中胚层 / 轴旁中胚层。 |
| SOM | 体节中胚层。 |
| LPM | 侧板中胚层。 |
| CP / FHF / SHF / JCF | 心脏新月 / 第一心区 / 第二心区 / 近心脏区。 |
| V-CM / IFT-CM / OFT-CM / A-CM / SV-CM | 心室 / 流入道 / 流出道 / 心房 / 静脉窦心肌。 |
| NCC | 神经嵴。 |
| Intra-Endoth / Chamber-Endo / BEC | 胚内内皮 / 心腔内膜 / 血管内皮。 |
| V-FG / D-FG / a-FG | 腹侧 / 背侧 / 前部前肠。 |
| SE / aSE / pSE / V-CSE / d-CSE | 表面外胚层及其分区。 |
| PAM-1…4 / Dorsal-PAM / Lateral-PAM | 轴旁中胚层亚型，E9.5 更名。 |
| ST / DMP / Proepi | 横膈相关间质 / dorsal mesenchymal protrusion / 前心外膜。 |
| Unknown | 未注释。插值保留经验比例，不要当出生簇放大。 |
