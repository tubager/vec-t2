# Task 1 技术方案（可执行）

- 对应总计划：[model_training_plan.md](model_training_plan.md)
- 官方任务页：[Temporal](https://virtualembryo.ai/challenge/tasks/temporal) · [评分](https://virtualembryo.ai/challenge/evaluation?section=scoring&task=1) · [数据](https://virtualembryo.ai/challenge/data)
- 本地评分：`pip install veckit`（当前 0.1.2）
- 文档日期：2026-08-25
- 术语：文末 [术语表](#术语表)；总计划里还有通用缩写。

本文只覆盖 **T1：心脏中心解剖的全转录组时间外推**。目标是：按下面的目录、命令和判定门，能从零跑通「地板基线 → 本地 E8.5→E9.5 评分 → 生成 E10.5 提交文件」。

---

## 1. 任务定义（用本地文件钉死）

给定已见阶段，生成**未来某一天、心脏中心解剖样本**的细胞群体（不是平均细胞、不是空间坐标）。

| 阶段 | 角色 | 本地文件 | 实测 |
|------|------|----------|------|
| E8.5 | 训练 | `data/E8.5_RNA.h5ad` | 16,787 细胞 × 32,285 基因 |
| E9.5 | 训练 | `data/E9.5_RNA.h5ad` | 17,057 细胞 × 32,285 基因 |
| E10.5 | 验证（P2/P3 前只给分） | 无 | 提交后打分 |
| E12.5 | 隐藏测试 | 无 | P3 起最多 2 次正式提交 |
| E7.75 | 全胚胎背景，**不是同一对象** | 本仓库暂无 | 可用作基因动力学旁证，不可当组成先验 |

两个训练文件的 `var_names` **完全同名同序**（已核验）。`.X` 为 float32、CSC 稀疏、`log1p`、非负；抽样最大约 6.0–6.6，零值约 88%。`obs` 只有 `celltype`。`obsm["X_umap.harmony.rna"]` 仅供可视化，**不要**写进提交。

官方补充：T1 是 **heart-centred dissection**，不是完整胚胎也不是纯心脏。随发育可解剖下来的周边组织在变。例如 Neural Tube 在 E8.5 占 4.60%，在 E9.5 为 0%。评分对比的是「带解剖偏差的目标群体」，模型不必复现解剖协议，但提交会被这种抽样打分。

### 1.1 提交契约

一份 AnnData `.h5ad`：

| 字段 | 要求 |
|------|------|
| `.X` | `float32`，`[n, 32285]`，log 归一化、有限、**非负**；稀疏或稠密均可 |
| `var.index` | 与榜单 panel **逐元素同序**。以官网 `T1__val.genes.txt` 为准，不要默认「训练文件的基因顺序一定等于榜单」 |
| `obs["celltype"]` | 可有，**评分忽略**。官方用冻结分类器重新打类型 |
| `obsm` | T1 **不要**空间坐标 |
| `n_obs` | 验证榜 **1,000–5,118**。推荐 **3,000**（MMD 最多抽 2,000，再多几乎无用） |

禁止：把平均细胞复制 n 次；把参考阶段按系数缩放当预测（DE 指标已对这种攻击做了 null）；交 MERFISH 的 500 基因。

### 1.2 评分（必须对这四项优化）

参照阶段：T1 是 **目标的前一可见阶段**（验证 E10.5 的 ref 是 E9.5；本地代理见第 4 节）。

| 题组 | 权重 | 指标 | 直觉 |
|------|------|------|------|
| DE 基因找回 | 25% | DES↑ | 上/下调基因集合对不对 |
| 变化方向 | 25% | DCS↑ | 全基因组 logFC 排序对不对（扣除高表达虚假相关） |
| 细胞状态分布 | 30% | MMD↓ | 类型与比例像不像同一群（目标 PCA-30 上的无偏 MMD） |
| 基因共变 | 20% | CSS↓ | 2 万对基因的 variogram 是否还在一起变 |

排行榜 skill：地板 `copy_last` = 50 分，天花板为答案对半切。**高于 50 才算学到变化。** 验证榜锚点（原始单位）：

| 指标 | 地板 | 天花板 |
|------|------|--------|
| `de_score` | 0 | 0.8464 |
| `de_direction` | 0 | 0.7901 |
| `mmd_u` | 0.08359 | 0.00406 |
| `variogram` | 0.005219 | 0.000158 |

### 1.3 为什么官方 Neural ODE 不够

E8.5 有 **18** 类，E9.5 有 **21** 类，**同名共享 11 类**。按细胞数：E9.5 里只有 **51.9%** 的细胞，其类型名在 E8.5 出现过；其余是新标签或新分化。速度场只能搬「还在的类型」，搬不动后半群体。本地标签集合：

**共享（11）**：AVC-CM, Blood, Foregut, IFT-CM, NCC, OFT/RV-CM, Pericardium, SV-CM, Surface Ectoderm, aSHF, pSHF

**仅 E8.5（7）**：EXEM, Endothelium, JCF, LV-CM, Neural Tube, Paraxial Mesoderm, RV-CM

**仅 E9.5（10）**：BEC, Endocardium, Epithelium, Hepatocyte, NCC-derived, Proepicardium, ST, V-CM, aPHM, pPHM

官方警告：词汇未对齐（E9.5 的 `Endocardium` 后面会拆成 `Endocardium-1/2`）。**不要把「标签集合差」当成生物学周转的全部。** 工作表示用表达 PCA / 软聚类，标签只用于配对和诊断。

---

## 2. 方法总览（三块，缺一不可）

```text
E8.5, E9.5
    │
    ├─ A. 组成 / 出生     类型（或 PCA 簇）占比如何随 t 变，允许降到 0、允许新簇
    ├─ B. 类型内运输      共享类型内部用非自治 OT-CFM，不要自治 Neural ODE
    └─ C. 残差 shift      官方 pseudobulk_shift 必留；深度模型只预测它解释不了的残差
    │
    ▼
3,000 × 32,285  非负 log1p 矩阵  →  E10.5 / E12.5 .h5ad
```

判定门：**本地 E8.5→E9.5 上，完整模型若全面弱于「经验重采样 E9.5 类型 + shift」或弱于 `pseudobulk_shift` 外推模板，提交只用简单件，不上 flow。**

不采用：端到端微调 scGPT / Geneformer 当提交模型（可作一周负对照）。T1 **允许**外部公开 scRNA，但 **禁止 MOCA 及任何 E9.5–E13.5 实测**（含 E10.5 / E12.5）。

---

## 3. 仓库布局（按此创建）

在项目根目录执行命令；原始数据在 `data/`：

```text
Embryo/
  data/
    E8.5_RNA.h5ad
    E9.5_RNA.h5ad
  doc/
    model_training_plan.md
    t1_technical_plan.md          ← 本文
  configs/t1.yaml
  panels/T1__val.genes.txt        ← 从官网下载，勿手改顺序
  src/t1/
    __init__.py
    io.py           读盘、对齐 panel、写提交
    pca.py          IncrementalPCA、编解码、clip≥0
    baselines.py    copy_last, pseudobulk_shift
    composition.py  类型占比、解剖衰减、出生簇
    ot.py           类型内 minibatch OT 配对
    flow.py         非自治 CFM
    losses.py       MMD / logFC / variogram（与 veckit 同构的可微近似）
    infer.py        指定目标 t 生成 n 个细胞
  scripts/
    00_audit.py
    01_baselines.py
    02_train_flow.py
    03_predict.py
    04_score_local.py
  outputs/t1/
    audit.json
    pca.joblib
    ckpt/
    preds/
```

Python 3.10–3.13。建议依赖：`anndata numpy scipy pandas scikit-learn torch veckit pyyaml joblib POT`（`POT` 做 OT；没有就用 scipy 的匈牙利在 PCA 上做小批量）。GPU 可选；CPU 也能跑 64 维 CFM。

`configs/t1.yaml` 建议初值：

```yaml
n_genes: 32285
n_submit: 3000
pca_dim: 64
pca_fit_cells: 8000          # 从两阶段各抽 4000 拟合 PCA
hvg: null                    # 第一版对全基因 IncrementalPCA；内存不够再改 3000 HVG
flow:
  hidden: [256, 256]
  lr: 1.0e-3
  batch_pairs: 256
  epochs: 200
  delta_t_unit: 1.0          # E8.5→E9.5 定义为 Δt=1
time:
  t_85: 8.5
  t_95: 9.5
  t_105: 10.5
  t_125: 12.5
shift:
  mode: "repeat_last_delta"  # E10.5: +1×Δ; E12.5: 先试 3×Δ，再网格 α∈{1,2,3}
  clip_min: 0.0
seed: 0
```

---

## 4. 本地评分协议（在没有 E10.5 答案时）

`veckit` **不是**真实验证分。它只保证格式和指标代码能跑。必须自己规定代理任务，否则无法选模型。

### 4.1 主代理：一步预测 E9.5（有答案）

- 输入：只用 E8.5 的细胞（训练 flow / 组成时 **可以**看 E9.5，因为这是唯一步观测；但要留 **类型 leave-out** 防过拟合标签）。
- 预测：生成「假 E9.5」。
- 评分：

```bash
veckit --task T1 \
  --input outputs/t1/preds/pred_E9.5.h5ad \
  --target data/E9.5_RNA.h5ad \
  --reference data/E8.5_RNA.h5ad
```

`--reference` 必须是 E8.5：DES/DCS 衡量的是相对前一阶段的**变化**。若省略，reference 会变成 input 自己，DE 项会假成 0。

地板对照（应接近官方「无变化」）：

```bash
# copy_last：用 E8.5 当预测去对 E9.5
veckit --task T1 --input data/E8.5_RNA.h5ad --target data/E9.5_RNA.h5ad --reference data/E8.5_RNA.h5ad
```

这一步的 `de_score` / `de_direction` 应对 0 附近（copy 没有变化）。MMD / variogram 则反映「隔一天群体差了多少」。

### 4.2 外推模板（无答案，只用于生成 E10.5 文件）

比赛上 `pseudobulk_shift` 预测 E10.5 的合理实现：

```text
Δ = mean(E9.5) − mean(E8.5)     # 长度 32285
X_E10.5 = clip(E9.5 + α·Δ, min=0)
α(E10.5) = 1
α(E12.5) ∈ {1, 2, 3}  用验证榜选，不要先验锁死 3
```

细胞从 E9.5 **子采样 3,000**（分层按类型更好），不要用全部 17,057（提交上限 5,118）。

### 4.3 不要做的「本地外推评分」

没有第三个 T1 时间点，**不能**在本地假装打 E10.5 的生物分。E7.75 是全胚胎，组成不可比。选模型只看：4.1 的一步分 + 提交后的官方 E10.5 分。P3 验证答案放出后再重训。

---

## 5. 第 0 周：审计与地板（必须先完成）

### 5.1 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install anndata numpy scipy pandas scikit-learn torch veckit pyyaml joblib
```

从官网下载 `T1__val.genes.txt` 放到 `panels/`。

### 5.2 `scripts/00_audit.py` 必须打印并写入 `outputs/t1/audit.json`

- `n_obs` / `n_vars`：16787 / 32285 与 17057 / 32285
- `var_names` 两文件一致；与 `T1__val.genes.txt` 逐行 equal（不等则后续全部 `reindex`）
- `X.min() >= 0`，无 NaN/Inf
- 类型表、共享/独有集合（应与第 1.3 节一致）
- Neural Tube 占 E8.5 约 4.60%、E9.5 为 0
- 写出 `outputs/t1/gene_order.npy`（提交一律按此顺序）

若 panel 与训练文件不一致：**训练仍用训练矩阵，写提交前按 panel 重排**，缺的基因填 0 并报警（T1 全转录组一般不应缺）。

### 5.3 `scripts/01_baselines.py`

实现两个函数，输出都是 3,000 细胞、panel 已对齐的 `.h5ad`：

**`copy_last(adata, n=3000, seed=0)`**  
分层或不分层子采样。预测 E9.5 时代理用 E8.5；预测 E10.5 用 E9.5。

**`pseudobulk_shift(src, ref_prev, ref_next, n=3000, alpha=1.0)`**  
`src` 是被平移的细胞（外推时为 E9.5），`delta = mean(ref_next) - mean(ref_prev)`。逐细胞加上 `alpha * delta`，`np.clip(..., 0, None)`，保持稀疏或转 dense 均可。

写提交示例：

```python
import anndata as ad
import numpy as np

def write_t1(X, var_names, path):
    # X: float32, (n, 32285), >=0
    adata = ad.AnnData(X.astype(np.float32))
    adata.var_names = var_names
    adata.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    adata.write_h5ad(path, compression="gzip")
```

跑：

```bash
python scripts/01_baselines.py
python scripts/04_score_local.py --pred outputs/t1/preds/copy_last_E95.h5ad
python scripts/04_score_local.py --pred outputs/t1/preds/shift_on_E85_to_E95.h5ad
```

其中 `shift_on_E85_to_E95` 若用了 E9.5 的均值，则 **泄漏了答案**，只能当「shift 类方法的上限诊断」，不能当模型选择依据。公平的一步 shift 诊断：在 E8.5 上加 **类型内** 噪声，或只用 E8.5 重采样，**不要**加真实 E9.5 均值。

公平一步基线（用于 4.1）：

1. `copy_last` = 子采样 E8.5  
2. `resample_types_then_noise` = 按 E8.5 类型经验分布采样 + 各基因独立小噪声（应输给任何有组成变化的模型）

外推提交基线（用于 4.2，交给官网）：

3. `copy_E95`  
4. `E95 + 1·Δ(E95−E85)` ← **P2 第一个官方提交就交这个**，作为线上地板

---

## 6. 模块 A：组成 / 出生

### 6.1 不要直接外推标签名

解剖消失与更名会让「线性外推 p(Neural Tube)」变成负比例。做法：

1. 把 E8.5/E9.5 的官方标签映射到 **工作簇**（下表，可改，改了要记进审计）。
2. 在工作簇上拟合 `p(k | t)`。
3. 推理时按 `p(k | t*)` 分层采样，再在簇内取表达。
4. 提交 **不写** `celltype`。

建议工作簇（第一版，手工，后续可用表达最近心自动合并）：

| 工作簇 | E8.5 标签 | E9.5 标签 | 备注 |
|--------|-----------|-----------|------|
| SHF | aSHF, pSHF | aSHF, pSHF | 第二心区，连续 |
| CM_OFT | OFT/RV-CM | OFT/RV-CM | |
| CM_IFT | IFT-CM | IFT-CM | |
| CM_AVC | AVC-CM | AVC-CM | |
| CM_SV | SV-CM | SV-CM | |
| CM_V | LV-CM, RV-CM | V-CM | 很可能是心室 CM 更名 |
| endo | Endothelium | Endocardium, BEC | 内皮/心内膜谱系 |
| peri | Pericardium | Pericardium, Proepicardium | 心包/前心外膜 |
| ncc | NCC | NCC, NCC-derived | |
| gut | Foregut | Foregut, Hepatocyte | 前肠/肝（解剖带进来的） |
| ect | Surface Ectoderm | Surface Ectoderm | |
| blood | Blood | Blood | |
| phm | — | aPHM, pPHM | E9.5 新出现，可能来自 SHF/中胚层 |
| st_epi | — | ST, Epithelium | 横膈/上皮，出生簇 |
| dissect_out | Neural Tube, Paraxial Mesoderm, EXEM, JCF | （E9.5 为 0） | **解剖丢失**，外推继续衰减到 0 |

`JCF`（juxtacardiac field）也可能并入 peri/SHF；第一版放 dissect_out，若 E10.5 线上 MMD 差再改并入 peri。

### 6.2 `p(k|t)` 的拟合

只有两个时间点，**不要**训大网络。对每个工作簇 k：

```text
logit π_k(t) = a_k + b_k · (t − 8.5)
π_k = softmax(logit)   # 或独立 sigmoid 再归一化
```

约束：

- `dissect_out`：强制 `b_k < 0`，且 `t≥9.5` 时 π=0  
- 出生簇（E8.5 为 0）：`π_k(8.5)=ε`，`π_k(9.5)=观测`，外推时 `π_k(10.5)=clip(π_k(9.5)·ρ, 0, π_max)`，ρ 默认 1.2，用线上 MMD 调  
- 总和为 1

实现：`src/t1/composition.py` 里 `fit_counts(df_stages) -> {k: (a,b)}` 与 `sample_cluster(t, n)`。

### 6.3 簇内表达库

每个 (阶段, 工作簇) 存该簇细胞的 PCA 坐标经验分布。生成时：

- 若目标 t 有「上一阶段同簇」：从该库采样，再走模块 B  
- 若是出生簇：从 E9.5 该簇采样，再加较大各向同性噪声 `σ_birth=0.5`（PCA 标准化单位）并走 B（Δt 用 1）  
- 若 dissect_out：不采样

---

## 7. 模块 B：PCA + 非自治 OT-CFM

### 7.1 PCA

`src/t1/pca.py`：

1. 两阶段各随机 4,000 细胞，`toarray()`，`sklearn.decomposition.IncrementalPCA(n_components=64)`。  
2. 保存 `mean_`、`components_`。  
3. 编码 `z = (x - mean) @ components.T`。  
4. 解码 `x = z @ components + mean`，`clip(x, 0, None)`。  

检查：在 500 细胞上重建 MSE 应远小于「用均值重建」。若重建把高表达基因抹平，DES 会差，则把 `pca_dim` 升到 96/128，或对 residual 再加一层按基因的线性（只对 HVG）。

**不要**在 UMAP 上做 flow。

### 7.2 类型内 OT 配对

对每个**共享工作簇**，取 E8.5 与 E9.5 的 z：

- 每步随机 `m=256` 对细胞  
- 在 z 空间算代价 `||z_i - z_j||^2`  
- `ot.emd` 或 `scipy.optimize.linear_sum_assignment` 得到配对 `(z0, z1)`  
- 记录 `Δt=1`，`t0=8.5`

出生簇不做 8.5→9.5 配对（没有源）。

### 7.3 速度场

```text
输入: concat(z, t_norm, dt, e_k)
  t_norm = (t − 8.5) / 4.0          # 覆盖到 E12.5
  dt     = Δt / 4.0
  e_k    = Embedding(K, 16)[cluster]
MLP: 64+1+1+16 → 256 → 256 → 64，SiLU
OT-CFM:
  τ ~ U(0,1)
  z_τ = (1-τ) z0 + τ z1
  u   = z1 - z0
  L_cfm = ||v_θ(z_τ, t0+τ·Δt, Δt, k) − u||^2
```

关键：**v 依赖绝对时间 t**（非自治）。推理 E9.5→E10.5 时传入 `t=9.5, dt=1`，不要把 E8.5→E9.5 的场原样积分两倍时间。

训练 200 epoch 即可。过拟合表现为：一步 E9.5 的 MMD 很好，但细胞挤成细丝（模式坍缩）——加 `σ=0.05` 的 z 噪声，或减小 hidden。

### 7.4 推理积分

从源细胞 z 出发，欧拉 10 步：

```text
for i in range(10):
    z ← z + v_θ(z, t + i/10·dt, dt, k) · (dt/10)
```

然后解码、clip。

---

## 8. 模块 C：残差 pseudobulk shift

在解码后的基因空间做，**不要**放进 PCA（DE 看的是全基因组均值）：

```text
x ← clip(x + α(t) · Δ, 0)
Δ = mean(E9.5) − mean(E8.5)
```

第一版让 flow **只学残差**：训练时把 `(z1 对应的 x1)` 换成 `x1 − Δ` 再编码，推理时再加回 `αΔ`。这样 CFM 不必重复学习全局平移（官方已证明平移很强）。

`α(10.5)=1` 固定。`α(12.5)` 等 P3 用验证集网格；P2 不要猜。

---

## 9. 损失（与指标同构，但可训练）

全部在 **子采样** 上算，对齐 scorer：细胞 ≤2000，基因对 20000。

| 损失 | 实现要点 | 对应 |
|------|----------|------|
| `L_cfm` | 第 7.3 节 | 运输 |
| `L_mmd` | 预测与目标 z 上 5 个 RBF 带宽的无偏 MMD；PCA 基 **拟合在目标上**（本地一步即 E9.5） | MMD 30% |
| `L_lfc` | `lfc_pred = mean(pred)−mean(ref)` vs `lfc_true`；`1 − Spearman` 或余弦 | DCS |
| `L_var` | 随机 4096 基因对，`mean((0.5·E|xi−xj|_pred − 0.5·E|xi−xj|_true)^2)` | CSS |

第一周只训 `L_cfm`。第二周 `L = L_cfm + 0.1 L_mmd`。`L_lfc` 很会把模型推向「只平移均值」、伤害 MMD，权重从 0.01 加起。若 DES/DCS 已由模块 C 覆盖，可以不加 `L_lfc`。

---

## 10. 生成 E10.5 / E12.5 的推理步骤

`scripts/03_predict.py --target 10.5 --n 3000 --out outputs/t1/preds/E10.5_v1.h5ad`

1. `π = composition.probs(t=10.5)`，丢掉 dissect_out。  
2. 按 π 分配各簇细胞数，凑满 3000。  
3. 每簇：从 **E9.5 该簇** 采样源 z（出生簇同）；`dt = t−9.5`（E10.5 为 1，E12.5 为 3）。  
4. 积分 v_θ（若该簇从未训练 flow：跳过积分，只加噪声）。  
5. 解码 + `α(t)·Δ` + clip。  
6. `var_names` 对齐 panel，写盘。  
7. 断言：`X.shape[1]==32285`，`X.min()>=0`，无 NaN，`n` 在 [1000, 5118]。

P2 提交顺序：

1. `E95 + 1·Δ`（线上地板）  
2. 组成重采样 E9.5（按外推 π）+ `1·Δ`，**无 flow**  
3. 仅当 4.1 上 flow 的 MMD **明显**好于「E9.5 重采样」时，再交带 flow 的版本  

每次只改一件事。记下提交 id 与四项原始指标。

---

## 11. 外部数据（可选，第 3 周以后）

| 可用 | 禁用 |
|------|------|
| Pijuan-Sala / Extended Mouse Atlas **E6.5–E9.5**，披露来源；基因取交集再填回 panel | MOCA；任何 E10.5、E12.5；E9.5–E13.5 窗口内数据；心脏 E10.5 atlas |

用途：给工作簇更多时间点，估 `b_k`。**不要**用来拟合 E10.5 组成的答案。标签需映射到工作簇；映射不了的细胞丢掉。

E7.75 全胚胎：只可看共享心脏相关簇的基因均值轨迹，**不要**用它的类型比例当 T1 组成。

---

## 12. 训练日程与门控

| 时间 | 产出 | 通过条件 |
|------|------|----------|
| 第 0 天 | audit.json、panel 一致、copy_last 的 veckit 能跑 | 基因顺序冲突为 0 |
| 第 1–2 天 | 线上提交 `E95+Δ` | 格式被收；记下四项分（预期 DES/DCS 尚可、MMD 一般） |
| 第 3–5 天 | 工作簇 + 组成外推 + 分层采样 + Δ | 本地一步 MMD 优于 copy_last；线上 MMD 优于提交 1 |
| 第 6–12 天 | PCA-64 + 簇内 CFM + 残差 Δ | 本地一步 MMD 优于「E9.5 经验重采样」才升上线；否则 **停用 flow** |
| 持续到 10-20 | 小步提交 | 只改 α、ρ、簇合并之一 |
| 10-20 后 | 用 E10.5 真值重训组成与 α | E12.5 只留 2 个正式提交：稳健（组成+Δ）与激进（+flow） |

算力：PCA + 200 epoch CFM 在单卡 12GB 或 CPU 数小时级。不必等大 GPU。

---

## 13. 关键代码骨架

### 13.1 公平的 copy_last / 外推 shift

```python
import numpy as np
import anndata as ad

def subsample(adata, n, seed, stratify=True):
    rng = np.random.default_rng(seed)
    if stratify and "celltype" in adata.obs:
        idx = []
        types = adata.obs["celltype"].astype(str)
        props = types.value_counts(normalize=True)
        for t, p in props.items():
            k = max(1, int(round(p * n)))
            cand = np.flatnonzero(types.values == t)
            take = rng.choice(cand, size=min(k, len(cand)), replace=len(cand) < k)
            idx.append(take)
        idx = np.concatenate(idx)[:n]
        if len(idx) < n:
            extra = rng.choice(adata.n_obs, size=n - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
    else:
        idx = rng.choice(adata.n_obs, size=n, replace=adata.n_obs < n)
    return adata[idx].copy()

def mean_X(adata):
    X = adata.X
    if hasattr(X, "mean"):
        mu = np.asarray(X.mean(axis=0)).ravel()
    else:
        mu = np.asarray(X).mean(axis=0)
    return mu.astype(np.float32)

def add_delta(adata, delta, alpha=1.0):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X = np.clip(X + alpha * delta, 0, None)
    out = ad.AnnData(X)
    out.var_names = adata.var_names
    out.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    return out
```

### 13.2 CFM 一步（示意）

```python
import torch
import torch.nn as nn

class Velocity(nn.Module):
    def __init__(self, d=64, k=16, n_clusters=16, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(n_clusters, k)
        self.net = nn.Sequential(
            nn.Linear(d + 2 + k, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, z, t, dt, cluster_id):
        e = self.emb(cluster_id)
        inp = torch.cat([z, t, dt, e], dim=-1)
        return self.net(inp)

def cfm_loss(model, z0, z1, t0, dt, cid):
    tau = torch.rand(z0.size(0), 1, device=z0.device)
    z_tau = (1 - tau) * z0 + tau * z1
    u = z1 - z0
    t = t0 + tau * dt
    v = model(z_tau, t, dt, cid)
    return ((v - u) ** 2).mean()
```

---

## 14. 失败模式（对照官方 pitfalls）

| 现象 | 原因 | 处理 |
|------|------|------|
| DES 很好、MMD 很差 | 只学会了均值平移 | 加组成采样与 MMD；禁止交平均细胞 |
| MMD 很好、DES≈0 | 群体像，但相对 E9.5 没动对基因 | 加回模块 C；检查是否不小心提交了 copy |
| 线上格式错误 gene mismatch | 没用 `T1__val.genes.txt` | 写盘前 `adata[:, panel]` |
| 负值被拒 | 解码或 shift 冲出 0 | 一律 clip |
| 本地一步极好、线上更差 | 过拟合 E9.5 标签名 | 用工作簇；少用 `L_lfc` |
| Endothelium 还在预测里大量出现 | 未做解剖衰减 | dissect_out 在 t≥9.5 置 0 |
| flow 不如 shift | 预期内 | **提交 shift**，不要为了用网络而用网络 |

---

## 15. 与总计划的关系

本文是 [model_training_plan.md](model_training_plan.md) 里 T1 的展开。T2 表达通道可复用这里的 PCA/组成/残差 shift，但基因是 MERFISH panel（胚胎验证 498、心脏 500），**不要**把 32,285 维提交接到 T2；T2 的几何通道、NFS 与三个榜单的判定门见 [t2_technical_plan.md](t2_technical_plan.md)。T3 不要用 T1 的 copy_last 思维（要用 WT→KO 的 delta）。

---

## 术语表

只列本文出现、且总计划未展开到「可对照本地文件」的词。通用缩写（MMD、OT-CFM、DES、veckit、held-out 等）见 [model_training_plan.md 术语表](model_training_plan.md#术语表)。

### 数据与提交

| 术语 | 含义 |
|------|------|
| heart-centred dissection | 以心脏为中心的解剖取样，会带上数量随阶段变化的周边组织（神经管、体节中胚层等）。 |
| panel / `T1__val.genes.txt` | 验证榜要求的 32,285 个基因符号，一行一个、顺序固定。提交必须逐元素对齐。 |
| CSC | Compressed Sparse Column，当前两个 RNA 文件的 `.X` 存储格式。 |
| `copy_last` | 把最近已见阶段子采样后原样提交。T1 地板。 |
| `pseudobulk_shift` | 每个细胞加上「两阶段平均表达之差」。T1 极强的简单基线。 |
| α | shift 的时间倍数。E10.5 取 1；E12.5 用验证集再选。 |
| IncrementalPCA | 分批主成分，避免 1.7 万×3.2 万稠密矩阵一次进内存。 |

### 细胞类型缩写（来自 `obs["celltype"]`）

| 标签 | 含义（工作理解） |
|------|------------------|
| aSHF / pSHF | anterior / posterior Second Heart Field，前/后第二心区。 |
| OFT/RV-CM | Outflow tract / right-ventricle cardiomyocytes，流出道/右室心肌。 |
| IFT-CM | Inflow tract cardiomyocytes，流入道心肌。 |
| AVC-CM | Atrioventricular canal cardiomyocytes，房室管心肌。 |
| LV-CM / RV-CM / SV-CM / V-CM | 左室 / 右室 / 静脉窦 / 心室心肌。E9.5 的 V-CM 很可能对应 E8.5 的 LV+RV。 |
| JCF | Juxtacardiac field，近心脏区。 |
| EXEM | Extra-embryonic mesoderm，胚外中胚层（解剖组成，不一定还在更晚的心脏取样里）。 |
| NCC / NCC-derived | Neural crest cells，神经嵴及其衍生。 |
| BEC | Blood endothelial cells，血管内皮（相对心内膜 Endocardium）。 |
| aPHM / pPHM | anterior / posterior pharyngeal mesoderm，咽中胚层（E9.5 新标签）。 |
| ST | Septum transversum，横膈相关间质（E9.5 新标签，需在 UMAP 上确认）。 |
| Endocardium | 心内膜。官方称后续阶段会拆成 Endocardium-1/2，勿与标签集合差画等号。 |

### 方法

| 术语 | 含义 |
|------|------|
| 工作簇 | 把两阶段不同标签名合并后的谱系桶，用来采样和 OT，不是提交标签。 |
| 出生簇 | E8.5 没有、E9.5 才有的工作簇；外推时从 E9.5 库采样再加噪声。 |
| dissect_out | 判定为「取样范围变了所以消失」的簇，外推比例打到 0。 |
| 非自治 | 速度场 v(z,t,Δt)，显式依赖发育时间，而不是只依赖状态的 v(z)。 |
| OT-CFM | 先用最优传输配对未对齐细胞，再在配对直线上回归速度。 |
| 残差 shift | flow 在「去掉全局均值差」的空间学位移，最后再把 Δ 加回。 |
| leave-out | 训练时丢掉某工作簇，检查是否只是在背标签名。 |
| 一步代理 | 用「由 E8.5 生成 E9.5」代替无法本地计分的 E10.5。 |

### 评分运算

| 术语 | 含义 |
|------|------|
| logFC | 基因在目标与参照之间的对数倍差，这里用两群体均值之差（log1p 空间）。 |
| Spearman | 秩相关。DCS 先对 logFC 做秩变换再偏相关。 |
| variogram（p=0.5） | 一对基因在细胞间绝对差的均值类统计，用来量共变。 |
| 无偏 MMD | 去掉核矩阵对角线的 MMD，完美预测时期望为 0 而不是 2/n。 |
| skill 50 | 与地板相同；T1 地板是 copy_last。 |
