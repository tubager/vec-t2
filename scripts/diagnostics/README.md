# 离线诊断（09-10 重做方案的证据）

这些脚本是 `doc/t2_p2_val_2026-09-05.md` §8 的证据来源。全部用**官方 veckit 指标**（`.venv/.../common/`），
在**有真值的窗口**上跑，不依赖「代理排名会迁移」的假设。跑法：`.venv/bin/python scripts/diagnostics/NN_*.py`。

| 脚本 | 回答的问题 | 结论 |
|------|-----------|------|
| `27_slice_gate.py` | 按簇发育进度切真细胞（全局 band 或空间分层）能否赢 mix 的 NFS、ODS 不崩、de 不崩？mmd 允许 ≤14% 税。 | 过门才许交原生几何。 |
| `26_flow_gate.py` | 留出中点上，左锚 CFM 是否同时赢 OT-place mix 的 NFS 和 mmd_u、且 de 不崩？ | 过门才许交。胚胎窗 E6.75+E8.0→E7.25；心脏 W2 E8.25+E9.5→E8.75。 |
| `21_expr_trajectory_estimators.py` | expr 组只取决于 `pb(pred)−pb(ref)` 的排序；把 `allowed_times` 之外的第三/第四个阶段补进来做二次、三次 Lagrange、局部线性回归，能否胜过现行两点线性 Δ？ | **否。** 三个窗口上两点线性最优或并列最优；「去掉 Δ 里的 ±pb(ref) 分量」也不涨。曲率在 0.25–0.75 天窗口内无可提取信号。 |
| `22_composition_estimators.py` | `CompositionT2.probs()` 是算术内插 `(1−w)pl+w·pr`；细胞群按增殖倍增，几何/对数线性/Hellinger/二次是否更好？ | **否。** 算术在 3/5 真值窗口最优，Hellinger 在 2/5 略优（−0.020 L1），几何普遍大幅更差（心脏代理 +0.32）。无一致赢家。 |
| `23_span_trim_validate.py` | `scripts/20_span_trim.py` 的 `auto` 门控在真值代理上是帮忙还是伤分？ | **门控成立。** 三个不需要它的窗口（心脏 238≤239、窄跨 173≤198）都正确 no-op；无条件的 union-all/intersect-all 在胚胎代理上伤 ODS（−0.011～−0.049），所以只有带目标占据数的 auto 版可交。 |
| `24_occupancy_census.py` | 各真实期与现役提交在规范 16³ 格上的占据体素数、径向分位。 | 心脏提交 nocc=289 vs 锚点 247/250（真值应≈248）→ 过度铺开 +17%，这是 ODS 43.3 低于地板的机制解释。 |

`23` 需要先造两个代理预测（真值分别为 E8.75 / E7.25）：

```bash
.venv/bin/python scripts/13_predict_t2.py --setting heart --target 8.75 --method full \
  --shape anisotropic --ot-interp --ot-x pick --ot-xyz left --ot-w 0.25 --n 5000 \
  --out /tmp/t2e/proxy/hrt_p875.h5ad
.venv/bin/python scripts/13_predict_t2.py --setting embryo --target 7.25 --method full \
  --shape tps --ot-interp --ot-x pick --ot-xyz left --n 5000 --ot-n-pair 6000 --ot-unique \
  --out /tmp/t2e/proxy/emb_p725.h5ad
```
