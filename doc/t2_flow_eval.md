# Task 2 OT-CFM 本地评估（2026-09-03）

- 对应方案：[t2_technical_plan.md](t2_technical_plan.md) §6.4 / 判定门
- 运行说明：[README.md](../README.md) Task 2 第 4–6 节
- 原始分表：[outputs/t2/eval_proxy.json](../outputs/t2/eval_proxy.json)
- 评分：`veckit` 本地代理（**不是**官网验证分）
- 训练：`python scripts/15_train_flow_t2.py`（默认 200 epoch，设备 MPS）
- 权重：`outputs/t2/embryo/ckpt/flow.pt`、`outputs/t2/heart/ckpt/flow.pt`

**结论先说：** 训练跑通、权重可用，但按判定门 **不要把** `--use-flow` **作为线上默认**。心脏外推上 flow 明确变差；胚胎上只比无 flow 的 `full` 好，MMD 仍打不过 `copy_last` / `shift_scale`。

---

## 1. 评估设置

每个 setting 单独训练 panel IncrementalPCA-32 + 非自治残差 OT-CFM，不复用 T1 的 32k 维网络。


| setting | 训练 hop                 | 代理任务                            | 对照                    |
| ------- | ---------------------- | ------------------------------- | --------------------- |
| 胚胎      | E6.75→E7.25、E7.25→E8.0 | 留出 E7.25：用 E6.75+E8.0 生成假 E7.25 | 真 E7.25，ref=E6.75     |
| 心脏      | E8.25→E8.75、E8.75→E9.5 | 一步外推：用 E8.75 生成假 E9.5           | 真 E9.5，ref=E8.75      |
| 心脏      | 同上                     | 生长诊断：E8.25→E8.75                | 真 E8.75，ref=E8.25     |
| 心脏      | —                      | E8.5 无真值，只查 RMS                 | 目标约 277，区间 (217, 354) |


对比方法：`copy_last`、`scale_copy`、`shift_scale`、`full`（组成+shift+按簇放点）、`full+flow`。n=3000。

指标方向：DES / DCS / ODS **越大越好**；MMD / CSS / SDD / NFS **越小越好**；TSR **越接近 0 越好**。判定门（方案原文）：本地 MMD 不明显好于「经验重采样 + shift」就停用 flow。

---



## 2. 训练日志摘要

PCA 重建好于均值基线，降维可用。


| setting | PCA recon MSE | 均值基线 MSE |
| ------- | ------------- | -------- |
| 胚胎      | 0.353         | 0.662    |
| 心脏      | 0.848         | 1.563    |


CFM loss 在 **4–6** 附近震荡，200 epoch 未压到接近 0。每个 epoch 只抽一个簇做一次 OT minibatch，有效更新偏少。

胚胎共享簇配对正常（出生簇 `neural` 未入流）。心脏 E8.75→E9.5 没有 `extra` / `nt`（FOV 丢失），出生簇 `CM_OFT` / `branch` / `st` 未入流，符合设计。E8.25 的 `extra` 约 3.1 万细胞、E8.75 只剩约 3.3 千，插值 hop 里该类极度不平衡。

---



## 3. 胚胎插值代理（假 E7.25 vs 真 E7.25）

与线上 E7.5 最同构。


| 方法            | DES↑       | DCS↑       | MMD↓        | CSS↓       | SDD↓       | ODS↑      | TSR→0  | NFS↓       |
| ------------- | ---------- | ---------- | ----------- | ---------- | ---------- | --------- | ------ | ---------- |
| copy_last     | 0.0345     | −0.0876    | **0.01855** | **0.0088** | 0.0586     | **0.863** | −0.186 | **0.0835** |
| scale_copy    | 0.0345     | −0.0876    | **0.01855** | **0.0088** | 0.0586     | **0.863** | +0.209 | **0.0835** |
| shift_scale   | 0.1724     | 0.3602     | 0.02442     | 0.1650     | 0.0586     | **0.863** | +0.209 | 0.1028     |
| full          | 0.2414     | 0.3519     | 0.05722     | 0.2780     | **0.0557** | 0.812     | +0.176 | 0.3086     |
| **full+flow** | **0.3103** | **0.4191** | 0.04814     | 0.2195     | 0.0572     | 0.852     | +0.177 | 0.2157     |


相对无 flow 的 `full`：flow 在 DES / DCS / MMD / CSS / NFS 上都更好。  
相对判定门：MMD **没有**赢过 `copy_last` 或 `shift_scale`。`full` 的 NFS 明显差于成对拷贝，按簇放点尚未接稳；flow 救了一部分，仍远差于 `copy_last`。

`scale_copy` 的 TSR 从 −0.19 翻到 +0.21，是用 E6.75 与 E8.0 做 log 线性插值偏大（E8.0 RMS 跳变），不是 flow 的问题。

预测文件：`outputs/t2/embryo/pred_E7.25_full_isotropic.h5ad`、`pred_E7.25_full_isotropic_flow.h5ad`。

---



## 4. 心脏一步外推代理（假 E9.5 vs 真 E9.5）

对应线上 E10.5 的技能。E9.5 标签几乎重标，MMD / NFS 预期会难看。


| 方法            | DES↑    | DCS↑    | MMD↓        | CSS↓       | SDD↓       | ODS↑      | TSR→0     | NFS↓       |
| ------------- | ------- | ------- | ----------- | ---------- | ---------- | --------- | --------- | ---------- |
| copy_last     | −0.0959 | 0.0558  | **0.05518** | **0.0552** | **0.0494** | 0.783     | −0.433    | **0.1215** |
| scale_copy    | −0.0959 | 0.0558  | **0.05518** | **0.0552** | **0.0494** | 0.783     | **0.000** | **0.1215** |
| shift_scale   | −0.1644 | −0.0827 | 0.07377     | 0.1476     | **0.0494** | 0.783     | **0.000** | 0.1535     |
| full          | −0.1781 | 0.0322  | 0.07251     | 0.1516     | 0.0600     | **0.790** | −0.003    | 0.2002     |
| **full+flow** | −0.1644 | 0.0837  | **0.10964** | 0.1984     | 0.0595     | 0.765     | −0.005    | 0.2414     |


flow 的 MMD / CSS / NFS **全面差于** 无 flow 的 `full`，更差于 `scale_copy`。外推不要开 flow。`scale_copy` 把 TSR 从 −0.43 收到 0，几何第一件成立。

预测文件：`outputs/t2/heart/pred_E9.5_full_isotropic.h5ad`、`pred_E9.5_full_isotropic_flow.h5ad`。

---



## 5. 心脏生长诊断（E8.25→真 E8.75）

类型名完全对齐，几何在缩小（FOV 收窄）。不要当成「心脏萎缩」的生物学结论。


| 方法            | DES↑      | DCS↑      | MMD↓       | CSS↓       | SDD↓       | ODS↑      | TSR→0     | NFS↓      |
| ------------- | --------- | --------- | ---------- | ---------- | ---------- | --------- | --------- | --------- |
| copy_last     | −0.272    | −0.042    | 0.0560     | 0.0543     | 0.0426     | 0.777     | +0.488    | **0.118** |
| scale_copy    | −0.272    | −0.042    | 0.0560     | 0.0543     | 0.0426     | 0.777     | **0.000** | **0.118** |
| full          | **0.826** | 0.933     | **0.0163** | **0.0493** | **0.0115** | 0.869     | +0.047    | 0.144     |
| **full+flow** | 0.772     | **0.942** | 0.0395     | 0.1010     | 0.0115     | **0.869** | +0.052    | 0.164     |


两侧对齐时，无 flow 的 `full` 已经很强。加上 flow 后 DES 0.83→0.77、MMD 0.016→0.039，变差。`scale_copy` 单独修好 TSR（+0.49→0）。

---



## 6. 心脏插值 RMS 门控（无 E8.5 真值）

目标 RMS 约 277（E8.25 与 E8.75 的 log 线性插值），合法区间 (216.9, 354.1)。


| 方法             | RMS       | |log(r/277)| | 是否在区间内 |
| -------------- | --------- | ------------ | ------ |
| **scale_copy** | **277.1** | **0.0005**   | 是      |
| full           | 255.6     | 0.0805       | 是      |
| full+flow      | 254.1     | 0.0861       | 是      |


心脏 E8.5 的尺度仍是 `scale_copy` 最准。`full` 略偏小（更靠近 E8.75）。

---



## 7. 判定与后续

1. **P2 不要默认提交** `--use-flow`**。** `configs/t2.yaml` 保持 `flow.use: false`。
2. 成对 `(X,xyz)` 之后：胚胎代理 NFS 0.31→0.11，心脏假 E9.5 NFS 已优于 `copy_last`；打乱 xyz 使 NFS 变差。
3. 各向异性：胚胎代理 SDD 0.059→0.030、ODS 0.833→0.862，过门。jitter 0/0.05/0.15 无差。TPS 不上（`shape.allow_tps: false`）。
4. **下一份线上文件**（无 flow、anisotropic `full`）在 `outputs/t2/submit/`：
   - `T2_embryo_val_interp.h5ad`（498，RMS 192）
   - `T2_heart_val_interp.h5ad`（500，RMS 277）
   - `T2_heart_val_extrap.h5ad`（500，RMS 335，β=0）
5. 心脏外推 β 只用官方 `scale_log_ratio` 反解，不要用 8.75→9.5 斜率。

复现评分（flow 仅作负对照，不要当提交）：

```bash
python scripts/13_predict_t2.py --setting embryo --target 7.25 --method full --use-flow
python scripts/13_predict_t2.py --setting heart --target 9.5 --method full --use-flow
python scripts/14_score_local_t2.py --proxy embryo_interp \
  --pred outputs/t2/embryo/pred_E7.25_full_isotropic_flow.h5ad
python scripts/14_score_local_t2.py --proxy heart_extrap \
  --pred outputs/t2/heart/pred_E9.5_full_isotropic_flow.h5ad
```

