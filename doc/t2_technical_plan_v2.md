# Task 2 技术方案 v2（P2 实证后）

- 取代：[t2_technical_plan.md](t2_technical_plan.md)（2026-09-02，从零跑通地板；**不要改那份**，当历史）
- 提交链：[t2_p2_val_2026-09-05.md](t2_p2_val_2026-09-05.md)
- 09-07 冲榜日记：[t2_p2_plan_2026-09-07.md](t2_p2_plan_2026-09-07.md)
- 09-09 提交队列：[t2_p2_plan_2026-09-09.md](t2_p2_plan_2026-09-09.md)
- 本地 flow 负对照：[t2_flow_eval.md](t2_flow_eval.md)
- 总计划：[model_training_plan.md](model_training_plan.md)
- 官方：[评分 Task 2](https://virtualembryo.ai/challenge/evaluation?section=scoring&task=2) · [数据](https://virtualembryo.ai/challenge/data) · [panel](https://virtualembryo.ai/challenge/panels/index.json)
- 文档日期：2026-09-07
- Python：`.venv/bin/python`；预测：`scripts/13_predict_t2.py`；本地代理：`scripts/14_score_local_t2.py`

本文只覆盖 **T2：3D MERFISH 时空预测**。两个 setting **分开拟合、分开提交、分数永不平均**。v1 解决的是「怎么从零交上合法文件」。v2 解决的是：**线上已经证明什么有效，三榜各用哪套配方，下一步只许动哪一块。**

---

## 0. 结论先说

当前最好（skill，50 = 官方 `copy_last` 地板）：

| 榜单 | skill | 配方 | n | RMS | 上传 |
|------|-------|------|---|-----|------|
| `T2:embryo:val_interp` | **66.32** | **OT-place + TPS**：真细胞 X + 形变左云，无 Δ | 3000 | **200** | `outputs/t2/submit/T2_embryo_val_interp.h5ad` |
| `T2:heart:val_interp` | **68.33** | **OT-place w=0.25**：真细胞 + 左锚 Hungarian，无 Δ | 5000 | **255** | `outputs/t2/submit/T2_heart_val_interp.h5ad` |
| `T2:heart:val_extrap` | **54.15** | `scale_copy`，β=**0.268**，拷 E9.5 | 5000 | **438** | `outputs/t2/submit/T2_heart_val_extrap.h5ad` |

平均 **62.93**。今日第 10 名 **64.03 = (69.5+56.5+66.1)/3**（还差 **1.10**）。心脏锁 **w=0.25** 且锁各向异性左锚（TPS 已否，SDD 63.6→43.4）。胚胎锁 **w=1/3**，TPS 形变可留（66.32）。禁止 lerp xyz。外推仍锁。

三榜不是同一个模型。共用代码路径 `generate()`，各自的数据、工作簇、组成、Δ、点云、n、RMS、时间权重。

第一名两个插值都在 **~73**，占的是目标阶段几何，不是缩尺左锚。心脏 ODS 仍 **45.3**（低于地板）。外推仍锁。

锁定：`flow.use: false`。`--shape tps` **只用于胚胎**（66.32）。心脏禁 TPS、禁右锚。外推不要加 α。不要 lerp xyz。

---

## 1. 任务定义（未变，用本地文件钉死）

给定已见阶段，生成**目标阶段的细胞群体**：每个细胞同时带 panel 表达和 `spatial_3D`。不是平均细胞，不是配准到固定图谱，也不是预测细胞数。

两个 setting 组织不同、坐标系不同。**不要把胚胎点云和心脏点云放进同一个生长模型。**

### 1.1 胚胎 setting（只有插值）

验证 / 隐藏测试都夹在训练区间内。决赛插值看隐藏 **E7.75**，不看心脏 E8.5。

| 阶段 | 角色 | 本地文件 | 实测 |
|------|------|----------|------|
| E6.75 | 训练 | `data/E6.75.h5ad` | 7,093 × **498**，18 类，RMS 122.7 |
| E7.25 | 训练（E7.5 的左锚） | `data/E7.25.h5ad` | 13,295 × **498**，18 类，RMS 147.3 |
| E7.5 | P2 验证插值 | 无 | 落在 7.25–8.0；时间权重 **w = 1/3** |
| E7.75 | 隐藏测试插值 | 无 | 同样夹在 7.25–8.0 |
| E8.0 | 训练（右锚） | `data/E8.0.h5ad` | 31,671 × **500**，26 类，RMS 326.6 |

验证 panel 是三者有序交集，**498 基因**。E8.0 多出的 `Casp4`、`Pnliprp1` 不能写进胚胎提交。E7.25 与 E6.75 类型名全共享；E8.0 相对 E7.25 按细胞数只有 **51.9%** 落在同名类（新标签含 FHF/SHF、前脑等）。

### 1.2 心脏 setting（插值与外推分开计分）

| 阶段 | 角色 | 本地文件 | 实测 |
|------|------|----------|------|
| E8.25 | 训练（插值左锚） | `data/E8.25_late.h5ad` | 58,716 × 500，33 类，RMS **354.1**（225.1 MB，完整） |
| E8.5 | P2 验证插值 | 无 | 落在 8.25–8.75；**w = 1/2** |
| E8.75 | 训练（插值右锚；T3 WT） | `data/E8.75.h5ad` | 24,826 × 500，33 类（与 E8.25 同名全集），RMS **216.9** |
| E9.5 | 训练（外推锚） | `data/E9.5.h5ad` | 53,742 × 500，22 类，RMS 335.0 |
| E10.5 | P2 验证外推 | 无 | 提交后打分 |
| E12.5 | 隐藏测试外推 | 无 | P3 起最多 2 次正式提交 |

心脏三文件 `var_names` 完全同序，且与胚胎 E8.0 的 500 基因同序。两个心脏验证榜 panel 相同。

**几何事实：** E8.25 RMS（354）**大于** E8.75（217），是取样 FOV 收窄，不是心脏在缩小。`copy_last` 拷 E8.25 的官方 TSR = **+0.328**（预测偏大）→ 插值必须**缩小**。外推 `copy_last` 拷 E9.5 的 TSR = **−0.268**（偏小）→ 放大。**禁止**用 E8.75→E9.5 的 log 斜率外推到 E10.5（会到 RMS≈598）；只用 E9.5 为锚、用线上 TSR 解 β。

E8.25 与 E8.75 类型名 100% 共享。E9.5 同名只剩 5 类；按细胞数只有 **32.3%** 落在 E8.75 同名类——词汇未对齐，不是细胞换了 70%。

### 1.3 矩阵与提交契约

`.X`：float32、log1p、非负、无 NaN。工作簇用 `obs["celltype"]`，不用几乎全是 `Unknown` 的 `cm_celltype`。坐标是每个胚胎局部系，阶段之间没有配准。提交只要 `spatial_3D`。

| 字段 | 要求 |
|------|------|
| `.X` | `float32` `[n, G]`，log 归一化、有限、**非负**。G 由该榜 panel 决定 |
| `var.index` | 与 panel **逐元素同序**。胚胎 498，心脏 500 |
| `obsm["spatial_3D"]` | `[n, 3]`；评分对平移/旋转不变；镜像是盲区 |
| `obs["celltype"]` | 可有，**评分忽略** |
| `n_obs` | 胚胎 **锁 3000**（合法 583–5000）；心脏插值 / 外推 **锁 5000**（合法上限更高） |

禁止：只交 `.X`；胚胎 500 基因；复制平均细胞；对绝对 xyz 做 MSE；把 n 当成组织大小；混用胚胎/心脏坐标。

### 1.4 评分（四组，必须对这四组优化）

参照 = 目标的前一可见阶段（E7.5→E7.25，E8.5→E8.25，E10.5→E9.5）。

| 题组 | 权重 | 指标 | 直觉 |
|------|------|------|------|
| 表达 | 25% | DES↑ 0.50 + DCS↑ 0.50 | 相对参照，哪些基因动了、方向对不对 |
| 细胞状态 | 25% | MMD↓ 0.60 + CSS↓ 0.40 | 类型/比例 + **基因共变** |
| 形状与尺度 | 25% | SDD / ODS / TSR 各 1/3 | 旋转不变形态 + 组织大小 |
| 局部空间 | 25% | NFS↓（kNN-15 邻域 pseudobulk 的 MMD） | `(X, xyz)` 必须联合对 |

排行榜 skill：地板 50、天花板 100。页面给的是 skill，不是原始 `scale_log_ratio`。每日 **8 次**；三榜分开交；**只记最好**（失败不拉低已记录最好）。

形状项：去质心 → PCA 主轴 → 除以自身 RMS。TSR 单独看 RMS 对数比。`det=+1` 四个翻转取最优。**不奖励镜像。**

验证榜原始锚点（`index.json`）：

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

ODS 天花板极窄（胚胎地板 0.705 → 天花板 0.766）。线上 ODS skill 掉到 40 表示比 `copy_last` 更差，不是「略差一点」。

### 1.5 外部数据

T2 不鼓励用外部 scRNA 补空间。保护窗：胚胎 E7.5–E7.75；心脏 E8.25–E8.75、E10.5、E12.5。更靠近 held-out 的外部数据视为作弊。

---

## 2. 当前成绩与缺口

对照（2026-09-07）：第 10 名 63.93 = (64.9+54.6+72.3)/3；第一名 68.93 = (73.4+59.8+73.6)/3。

| 榜单 | 我们 | vs 10th | vs 1st | 短板 | 已打满 |
|------|------|---------|--------|------|--------|
| 心脏插值 | **65.31** | **+0.4** | **−8.1** | CSS 55.3，NFS 63.6 | TSR 100，SDD **72.6** |
| 心脏外推 | 54.15 | −0.5 | −5.7 | 表达停在拷贝（DES 45.8） | TSR 100 |
| 胚胎插值 | 66.18 | −6.1 | **−7.4** | ODS 55.6（曾 40），DES 57 | TSR 98.8，CSS 67.5 |

题组约数：

| 题组 | 胚胎 66.18 | 心脏插值 65.31 | 外推 54.15 |
|------|------------|----------------|-----------|
| 表达 25% | ~64 | **66** | **48** |
| 状态 25% | **68**（CSS 67.5） | **58**（CSS 55.3） | 49 |
| 形状 25% | ~70（ODS 55.6） | **73**（SDD 72.6） | 66（SDD 44） |
| NFS 25% | 62.7 | 63.6 | **51** |

胚胎 66.18 分项：DES 57.0、DCS 70.2、MMD 69.0、CSS 67.5、SDD 55.7、ODS 55.6、TSR 98.8、NFS 62.7。  
心脏 65.31：DES 64.3、DCS 68.4、MMD 60.1、CSS 55.3、SDD 72.6、ODS 46.8、TSR 100、NFS 63.6。

心脏插值已过第 10 名的 64.9。对第一名仍差 ~8。NFS 收回是近路。外推继续锁。

---

## 3. 从线上学到的设计原则

这八条是 v2 的方法约束，优先级高于 v1 里「组成 + shift + 按簇放点」的默认叙事。

1. **CSS 要真细胞，不要均值平移，不要基因 lerp。**  
   全局 Δ 把均值推过去，共变停在源阶段：胚胎 `shift_scale` CSS 40.4，心脏 33.6。成对 mix **无 Δ** 把胚胎 CSS 打到 66.9。OT 对配对细胞做 `(1−w)X_L+w X_R` 把 CSS 冲回 58.7、MMD 冲回 55.6（62.46，回退）。

2. **一团几何，不要 50:50 叠两侧点云。**  
   成对 mix 无 Δ：胚胎 ODS 58→**39.7**（65.12 仍过门是因为 CSS/MMD）；心脏 w=1/2 叠云：CSS 41→56.4，但 SDD 62→**38**、NFS 70→59（61.76，回退）。心脏比胚胎更伤，因为插值权重是一半对一半；胚胎 mix 其实是 **2:1 偏左**。

3. **右锚表达可以放进提交，xyz 必须接到匹配的左锚位置，不能随机贴。**  
   同一份 mix X，`--mix-xyz left` 随机贴：表达四项不变，ODS 40→51，NFS 64→**59.5**，总分 64.63，差 0.5 不过门。OT Hungarian 按表达配对后再贴：ODS 55.6，NFS 只掉到 62.7，总分 **66.18**。

4. **几何锚锁最近左锚。**  
   心脏 `--geom-src right`（E8.75 放大到 RMS 255）：ODS 48→55，SDD 62→**31**、NFS 70→**42**，总分 53.08。禁止右锚几何。

5. **n 不是越大越好，且两榜相反。**  
   胚胎 3000→5000：ODS 58→41，60.90 回退。心脏 3000→5000：+1.4（ODS 本就在地板以下）。外推 3000→5000：54.15 过门（TSR 仍 100，SDD 略掉）。

6. **TSR 用线上反推的 RMS，不要用 log 线性插值当最终尺度。**  
   胚胎 log 线性 ≈192，线上锁 **200**（TSR 98.8）。心脏插值 log 线性 ≈277，线上锁 **255**（TSR 100）。外推 β=**0.268**、RMS **438**（TSR 100）。生成后各向同性再缩到这些常数。

7. **本地 `occupancy_dice` 不能预测线上 ODS。**  
   成对 mix 本地 ODS 0.87，线上 39.7。代理只用来挡 CSS/MMD/NFS 的方向性灾难（lerp 表达、随机贴点），过门仍看线上总分。

8. **Flow / TPS / 外推 shift 本地或线上已否。**  
   panel OT-CFM 胚胎 MMD 打不过 `copy_last`/`shift_scale`；心脏外推 flow 全面变差。外推 `shift_scale` 线上 42.5。外推无 Δ 的 `full` 53.14，拆开 E9.5 成对几何伤 SDD。

---

## 4. 三榜生产配方（当前默认）

代码：`src/t2/infer.py` 的 `generate()`；OT 放置：`src/t2/ot.py` 的 `ot_interpolate_alloc(..., x_mode="pick", xyz_mode="left")`。配置：`configs/t2.yaml`。

写盘后**必须**再缩到下表 RMS（`rescale_after_place` 缩的是 log 线性目标，和线上锁定量不一致）。

```python
import numpy as np, anndata as ad
from pathlib import Path

def rescale_rms(src: str, dst: str, r_target: float) -> None:
    a = ad.read_h5ad(src)
    xyz = np.asarray(a.obsm["spatial_3D"], dtype=np.float64)
    X = xyz - xyz.mean(0)
    r = float(np.sqrt((X**2).sum(1).mean()))
    a.obsm["spatial_3D"] = (X * (r_target / r)).astype(np.float32)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(dst, compression="gzip")
```

### 4.1 胚胎插值（默认 = 当前最好 66.32）

簇内在 panel PCA-32（`outputs/t2/embryo/pca.joblib`）做 Hungarian；以 w=1/3 Bernoulli 抽**真细胞**（左或右），**X 原样**；xyz 用配对左锚坐标，再按时间权重把**整团左云** OT 形变（`--shape tps`）；出生簇从右锚拷；无全局 Δ；最终 RMS 200。

```bash
.venv/bin/python scripts/13_predict_t2.py --setting embryo --target 7.5 --method full \
  --shape tps --ot-interp --ot-x pick --ot-xyz left --n 3000 \
  --out outputs/t2/embryo/pred_E7.5_full_tps_otinterp_xpick_xyzleft.h5ad
# 再 rescale_rms(..., r_target=200) →
# outputs/t2/embryo/pred_E7.5_full_tps_otinterp_xpick_xyzleft_rms200.h5ad
```

对照换最好：总分 **> 66.32**；CSS 不得掉到 60 以下；NFS 不得掉到 60 以下。不要退回 lerp xyz。

### 4.2 心脏插值（默认 = 当前最好 65.31）

OT-place 与胚胎相同，几何锁 **E8.25**，时间权重默认 w=**0.5**，n=5000、RMS 255。不要 mix、不要右锚。

```bash
.venv/bin/python scripts/13_predict_t2.py --setting heart --target 8.5 --method full \
  --ot-interp --ot-x pick --ot-xyz left --n 5000 \
  --out outputs/t2/heart/pred_E8.5_full_anisotropic_otinterp_xpick_xyzleft_n5000.h5ad
# rescale_rms(..., 255)
```

对照换最好：总分 **> 65.31**；SDD ≥ 65；NFS 相对 63.6 不得再大掉。下一格候选 `--ot-w 0.25`（少混右锚，换 NFS）。

若过门且 NFS 微掉：下一格只改混合比（把 `w` 从 0.5 降到 ~0.25），几何仍左锚。不要再交成对叠云 mix。

### 4.3 心脏外推（默认 = 当前最好 54.15，锁住）

拷 E9.5 表达和点云，各向同性放大到 RMS 438（β=0.268）。不要组成拆云，不要 Δ。

源文件：`outputs/t2/heart/scale_copy_E105_beta0268_n5000.h5ad`。配置键 `heart_extrap_beta_105: 0.268`。过门还要求 TSR skill **= 100**。

### 4.4 目录纪律

| 位置 | 放什么 |
|------|--------|
| `outputs/t2/submit/` | 仅当前三份最好 + `manifest.json` |
| `outputs/t2/embryo/`、`heart/` | 当前最好的源 h5ad + 已排队未交的一份；以及 `pca.joblib`、`composition.json`、ckpt |
| `outputs/t2/archive/YYYY-MM-DD/` | 代理、未缩尺副本、已否 / 被替代的预测 |

不要交：`archive/`、`*_flow.h5ad`、`--shape tps`、代理 `pred_E7.25_*` / `*_proxy_*` / `*_growth_*`。

---

## 5. 方法（按通道，以实证为准）

两 setting 仍分模型。v1 的三块还在，但插值的联合通道已从「按簇随机放点」升级为 **OT-place**。

```text
setting ∈ {embryo, heart}     ← 永不混坐标系
    │
    ├─ A. 表达   组成 π(t*) + 真细胞库（插值 OT-place / 心脏 full 的 Δ）
    ├─ B. 几何   左锚点云 × 各向异性轴比 × 线上 RMS
    └─ C. 联合   簇内 Hungarian：X 保持真细胞，xyz 用匹配左锚
    │
    ▼
n × G 的 .X  +  spatial_3D
```

### 5.1 表达

工作簇仍按 v1 映射（`src/t2/clusters.py`）。标签名不外推：映射到工作簇 → 插值/冻结 π → 簇内采样。提交不依赖 `obs["celltype"]`。

**插值组成：**

```text
π(t*) = (1 − w) π(t_left) + w π(t_right)
w_embryo_E7.5 = 1/3
w_heart_E8.5  = 1/2
```

出生簇（胚胎 `neural`；心脏 `CM_OFT` / `branch` / `st`）从右锚采样。Unknown 保留经验比例，不当出生簇放大。

**外推组成：** 冻结 π(E9.5)。FOV 丢失簇（EXE/尿囊等）在 t≥9.5 保持 0。当前提交甚至不走组成——直接 `scale_copy` E9.5。

**Δ 的用法（已证伪的默认）：**  
v1 把「左锚细胞 + wΔ」当插值主力。线上：DES/DCS 大涨，CSS 大跌。心脏 `full` 仍带按簇 Δ，是 64.58 的 DES 来源，也是 CSS 洞。胚胎 OT-place **强制 no_delta**。混库后再加 Δ，本地 CSS/MMD/NFS 全面差于无 Δ。

**OT-place 的 X：**

```text
簇内 Hungarian（panel PCA）得到配对 (i_left, i_right)
n_right ~ Binomial(n_cluster, w)
抽到右：X = X_right[i_right]     xyz = xyz_left[i_left]
抽到左：X = X_left[i_left]      xyz = xyz_left[i_left]
```

`--ot-x lerp` 禁止上线。`--ot-xyz lerp` 本地 ODS 0.77，禁止上线。

可选 flow：`scripts/15_train_flow_t2.py` 权重可用，**提交不开**。见 [t2_flow_eval.md](t2_flow_eval.md)。

心脏外推不要投影 T1 Δ（`t1_delta_weight: 0`）。胚胎不要用 T1 心脏解剖组成。

### 5.2 几何

规范化（建模用）：去质心、RMS、PCA 主轴。提交坐标系任意。禁止 `||xyz_pred − xyz_true||^2`。

实测 RMS / 主轴 std：

| 阶段 | RMS | 主轴 std |
|------|-----|----------|
| E6.75 | 122.7 | 100.9, 52.2, 46.3 |
| E7.25 | 147.3 | 105.3, 86.7, 55.6 |
| E8.0 | 326.6 | 226.6, 202.1, 120.3 |
| E8.25 | 354.1 | 245.3, 210.4, 144.9 |
| E8.75 | 216.9 | 136.8, 128.3, 108.9 |
| E9.5 | 335.0 | 244.1, 178.3, 144.4 |

**尺度（线上锁定，覆盖 log 线性）：**

| 目标 | 生成时 log 线性 | 提交 RMS | TSR skill |
|------|-----------------|----------|-----------|
| 胚胎 E7.5 | ≈192 | **200** | 98.8 |
| 心脏 E8.5 | ≈277 | **255** | 100 |
| 心脏 E10.5 | β 未用斜率 | **438**（β=0.268） | 100 |

各向异性：插值对三主轴 std 做 log 线性，再整体缩到上表 RMS。心脏外推轴比等于 E9.5（等价各向同性缩放）。心脏不要改回各向同性（会吐掉 SDD ~62）。TPS / 点云 OT 插值：`allow_tps: false`。

镜像是盲区，不为 laterality 优化。

### 5.3 NFS 联合

NFS 要的是「这种表达周围是那种表达」，不要求细胞一一对应。

已否：

- 从整胚 blob 随机配 xyz（邻域被抹平）
- 右锚 X + 随机左锚 xyz（NFS −4.5）
- 左锚 X + 右锚几何（心脏 NFS 70→42）
- 两团成对云（心脏 NFS 70→59；胚胎 ODS 崩）

有效：OT 匹配后的一团左锚。配对用 PCA 空间平方欧氏距离、Hungarian、每簇最多 `n_pair=800`。抖动 `jitter_frac=0.15`（相对中位 kNN）。本地诊断：同一份 X 打乱 xyz，NFS 应变差。

---

## 6. 工作簇（与 v1 相同，改映射要记 `audit.json`）

### 6.1 胚胎

E6.75 / E7.25 十八类同名。E8.0 二十六类。

| 工作簇 | E6.75 / E7.25 | E8.0 | 备注 |
|--------|---------------|------|------|
| exe_endo | EXE-Endoderm | EXE-Endoderm | |
| exem | ExEM-1, ExEM-2 | ExEM-1, ExEM-2, p-EXEM | |
| exe_ect | EXE-Ectoderm | — | E8.0 为 0 |
| epi | Anterior Epiblast, Caudal Epiblast | Caudal Epiblast | |
| streak | Primitive Streak | Primitive Streak | |
| gut | Gut Endoderm | V-FG, D-FG | |
| se | aSE, pSE | V-SE, pSE | |
| phm | PHM/PAM | aPHM, pPHM | |
| som | SOM | SOM | |
| lpm | LPM | LPM | |
| allantois | Allantois | Allantois, Allantois Endothelium | |
| hem | HEM-Endoth, Blood Progenitor | HEM-Endoth, Intra-Endoth, EXE-Endothlium | |
| heart | CP | FHF, SHF, JCF | |
| neural | — | Forebrain, Hindbrain, d-CSE | E8.0 出生 |
| unknown | Unknown | Unknown | 插值保留比例 |

### 6.2 心脏

E8.25 / E8.75 三十三类同名。E9.5 二十二类，同名 5 个。

| 工作簇 | E8.25 / E8.75 | E9.5 | 备注 |
|--------|---------------|------|------|
| CM_V | V-CM | V-CM | |
| CM_IFT | IFT-CM | A-CM, SV-CM | |
| CM_OFT | — | OFT-CM | 出生 |
| endo | Endo, Intra-Endoth-1/2 | Chamber-Endo, Cush-EndoMT-Endo, BEC, early-VEC, Great Artery Endoth | |
| peri | Peri | Peri, Proepi | |
| ncc | NCC | NCC | |
| phm | aPHM, pPHM | aPHM, pPHM | |
| pam | PAM-1…4 | Dorsal-PAM, Lateral-PAM | |
| se | pSE, V-CSE, d-CSE | SE | |
| gut | D-FG, a-FG, Lateral FG, Gut Endoderm, Hindgut | Foregut, Hepatocytes | |
| nt | Neural Tube, Forebrain | — | 外推衰减，不线性外推 15% |
| extra | EXE-Endoderm, ExEM, Allantois, HEM-Endoth, SOM, LPM, Caudal Epiblast, JCF, Unknown | — | FOV 丢失，t≥9.5 为 0 |
| branch | — | Branch Arch | 出生 |
| st | — | ST, DMP | 出生 |

---

## 7. 本地代理（选模型用，不是官网分）

`veckit` 只保证格式和指标代码。没有验证答案时必须用代理，但 **ODS 代理不可信**，过门以线上总分为准。

```bash
.venv/bin/python scripts/14_score_local_t2.py --proxy embryo_interp \
  --pred path/to/pred_E7.25_....h5ad
```

| 代理 | 怎么做 | 用途 | 不要当什么 |
|------|--------|------|------------|
| 胚胎插值 | 留出 E7.25，用 E6.75+E8.0 生成假 E7.25，ref=E6.75，w=0.4 | 与 E7.5 同构；挡 lerp / 随机贴点 | 线上 ODS |
| 心脏一步外推 | 用 E8.75 生成假 E9.5，ref=E8.75 | 外推技能；标签重标，MMD/NFS 会难看 | 线上 E10.5 的 DES 保证 |
| 心脏生长诊断 | E8.25→真 E8.75 | 确认 TSR 符号、类型对齐时 `full` 很强 | 「心脏在萎缩」 |
| 心脏 E8.5 | **无真值** | 只查 RMS ∈ (216.9, 354.1) | 生物分 |

胚胎 OT-place 假 E7.25（原始单位；MMD/CSS/NFS/SDD ↓ 越好，DES/ODS ↑ 越好）：

| 指标 | 成对 mix 无 Δ | mix 随机左锚 | **OT-place** | OT lerp X |
|------|---------------|--------------|--------------|-----------|
| MMD↓ | 0.018 | 0.0177 | **0.0173** | 0.033 |
| CSS↓ | 0.030 | 0.0299 | **0.0302** | 0.072 |
| ODS↑ | 0.871 | 0.863 | 0.863 | 0.855 |
| NFS↓ | **0.080** | 0.114 | **0.090** | 0.097 |
| SDD↓ | — | 0.0306 | **0.0269** | — |

本地门（占格子前）：OT-place 的 CSS/MMD 必须与 mix 无 Δ 同档（真细胞）；NFS 必须明显好于随机左锚。lerp 表达或 NFS≥随机贴点 → 不上线。

没有第三个心脏时间点在 E9.5 之后，**不能**本地假装打 E10.5。

---

## 8. 锁定、禁止、过门

### 8.1 锁定

- 胚胎 RMS **200**、**n=3000**
- 心脏插值 RMS **255**、n=**5000**、几何 **E8.25**
- 心脏外推 n=**5000**、β=**0.268**、RMS **438**
- `flow.use: false`，`shape.allow_tps: false`，`t1_delta_weight: 0`

### 8.2 禁止再交（线上或本地已否）

| 实验 | 结果 | 教训 |
|------|------|------|
| 外推 α=1 `shift_scale` | 53.3→**42.5** | 外推不要加 Δ |
| 外推无 Δ `full` | **53.14** | 不要拆 E9.5 成对几何 |
| 胚胎 mix 未修 xyz | **60.18** | 禁止随机拆对 |
| 胚胎 n=5000 `full` | **60.90** | 胚胎锁 3000 |
| 胚胎 OT lerp X | **62.46** | 禁止 `--ot-x lerp` |
| 心脏 `--geom-src right` | **53.08** | 禁止右锚几何 |
| 心脏成对 mix 无 Δ | **61.76** | 禁止叠云；CSS 涨不够补 SDD |
| 胚胎 mix `--xyz left` 随机 | **64.63** | 一团对，配对必须 OT |
| `--use-flow` / TPS | 本地未过门 | 保持关 |
| 心脏改回各向同性 | 未交，已判净亏 | 保 SDD ~62 |
| `--dens-keep` | 本地占位/SDD 变差 | 禁止 kNN 核截尾 |
| 丢掉表达库的 FOV 截 | 本地 DES 变差 | 只许同 X 的 `--crop-xyz` |
| `--ot-xyz morph` / `occ` | 本地输给 TPS / 涂抹解剖 | 禁止再插值 xyz |

不要为救 CSS 加大 α。不要把三榜当同一个模型调。不要 8 次都加 n。不要把各格「期望」高档加总当进前十的路径。

### 8.3 换最好

- 该榜总分必须更高。
- 心脏插值：SDD 掉到 55 以下或 NFS 掉到 65 以下，即使 CSS 涨也回退。
- 外推：TSR 必须仍为 100。
- 胚胎：CSS < 60 或 NFS < 60 回退。

出分后记八项 skill + 总分，更新 [t2_p2_val_2026-09-05.md](t2_p2_val_2026-09-05.md) 与 `outputs/t2/submit/manifest.json`。过门才把源文件拷进 `submit/`。

---

## 9. 仓库与复现

```text
Embryo/
  data/          E6.75 E7.25 E8.0；E8.25_late E8.75 E9.5
  panels/        三个 T2 panel，勿手改顺序
  configs/t2.yaml
  src/t2/        io, clusters, geometry, growth, shape, neighborhood,
                 expression, flow, pca, ot, infer
  scripts/       10_audit 11_baselines 12_fit_geometry
                 13_predict 14_score_local 15_train_flow
  outputs/t2/    embryo/ heart/ submit/ archive/
  doc/           本文；v1 历史方案；val log；flow eval
```

Python 3.10–3.13。几何是 numpy/scipy，不需要 GPU。Hungarian 每簇 ≤800，CPU 秒级。

从零审计（已完成，不必重做除非数据更新）：

```bash
.venv/bin/python scripts/10_audit_t2.py
```

应与 §1 一致：细胞/基因数、胚胎 498 = E8.0 去掉 `Casp4`/`Pnliprp1`、三 panel 逐行 equal、`X.min()>=0`、`spatial_3D` 为 `(n,3)`、E8.25 ≈ 225 MB。

---

## 10. 下一步（09-08 起）

每日 8 次，每次只改一件事。试坏不影响已记录最好。

1. **停拧 w。** 心脏 0.25、胚胎 1/3 已锁。0.20 / 0.40 / 0.50 / 0.25（胚胎）均否。
2. **不要交 `--dens-keep`。** 心脏生长代理上占位/SDD/NFS 全差于未截尾 `scale_copy`。
3. 几何插值已走完（lerp xyz / TPS / morph / occ）。左云 **FOV 半径截**改为只换 xyz、X 与 68.33 同一份：`--crop-to-rms --crop-rms 255`。E8.25 58,716→33,548。未写入 `submit/`。不要交丢掉表达库的旧 `--crop-to-rms`（DES 诊断变差）。
4. 外推仍锁。胚胎不要跟着重截，除非心脏线上 ODS 明显涨且 CSS/NFS 保住。

不要再交：lerp 表达、右锚几何、成对叠云、胚胎 n=5000、外推 α、flow、心脏 TPS、lerp xyz、morph、occ、dens-keep。

现实落点：FOV 截若把心脏 ODS 45→60 且 CSS 保住，该榜约 +1，平均 ~63.3，仍可能不到 64 或 68.9。两边都到第一名插值档（~73）没有已知配方。

---

## 11. 失败模式（官方 pitfalls + P2 实测）

| 现象 | 原因 | 处理 |
|------|------|------|
| 胚胎被拒 gene mismatch | 500 基因或用了 E8.0 顺序 | 对齐 `T2__embryo__val_interp.genes.txt` |
| 缺 `spatial_3D` | 只写了 `.X` | `write_t2` 必须写 obsm |
| 心脏插值 TSR 大正数 | 把 E8.25 放大了 | 提交 RMS **255** |
| 外推 TSR 过冲 | 用了 8.75→9.5 斜率（RMS≈598） | β=0.268，RMS 438 |
| DES 好、CSS 差 | 只做了全局 Δ | 真细胞，不要加大 α |
| CSS 好、ODS 崩 | 叠了两团点云 | OT-place 一团左锚 |
| ODS 回升、NFS 掉、总分平 | 右锚 X 随机贴左锚 | Hungarian，不要 `--mix-xyz left` |
| CSS/MMD 掉、ODS 回升 | 基因 lerp | `--ot-x pick`，不要 lerp |
| SDD/NFS 崩、ODS 涨 | 心脏改用 E8.75 几何 | 锁 E8.25 |
| 胚胎加 n、ODS 崩 | 占位铺开 | n=3000 |
| 本地 ODS 很高、线上 40 | 代理不可信 | 以线上总分为准 |
| flow 不如 shift / copy | 预期内 | 提交不开 flow |
| 对 xyz 做 MSE 形状仍差 | 评分忽略刚体 | 删坐标 MSE |
| 负值 | shift / 解码冲出 0 | clip ≥ 0 |
| 镜像胚胎 | 盲区 | 不优化 laterality |

---

## 12. 与 v1、T1、总计划的关系

- **v1**（[t2_technical_plan.md](t2_technical_plan.md)）：任务定义、panel、工作簇、地板基线、几何规范化仍有效。默认推理「shift + 按簇放点 + 可选 flow」已被 P2 证伪或降级。
- **T1**（[t1_technical_plan.md](t1_technical_plan.md)）：组成 / 残差 / PCA 的思路可借鉴；**禁止**把 32,285 维接到 T2。心脏外推不投影 T1 Δ。
- **T3**：不要用本文的 `copy_last` 思维（要用 WT→KO 的 delta；形状不计入 T3 总分）。
- 验证日记与冲榜表继续写在 val log / 09-07 plan；本文是稳定配方，不按天追加「第 n 次」。

---

## 术语表

通用缩写（MMD、OT-CFM、DES、veckit、held-out、Procrustes）见 [model_training_plan.md 术语表](model_training_plan.md#术语表)。细胞类型缩写见 v1 术语表。

| 术语 | 含义 |
|------|------|
| setting | `embryo` 与 `heart`。分数不平均、模型不共享。 |
| skill | 官网分。50=`copy_last`，100=天花板。 |
| OT-place | `--ot-interp --ot-x pick --ot-xyz left`。真细胞 + 匹配左锚坐标。 |
| 成对 mix | `--mix-expr` 且 xyz 跟表达同源。两团云，胚胎曾 65.12。 |
| 叠云 | 左右锚点云都进提交。心脏 w=1/2 时尤其伤 SDD。 |
| FOV | 取样视野。心脏 E8.25→E8.75 RMS 变小主要是 FOV，不是器官萎缩。 |
| β | 加在 `log r(E9.5)` 上的外推步长。锁定 0.268。 |
| RMS radius | 到质心距离的均方根。TSR 用它当尺度。 |
| SDD / ODS / TSR / NFS | 形状分布 EMD / 占据 Dice / 尺度对数比 / 邻域 MMD。 |
| laterality | 左右不对称。形状项对镜像同样给分。 |
| 498-gene intersection | 胚胎 panel。缺 `Casp4`、`Pnliprp1`。 |
| `E8.25_late.h5ad` | 心脏 E8.25 训练文件，58,716 细胞、225.1 MB。 |
