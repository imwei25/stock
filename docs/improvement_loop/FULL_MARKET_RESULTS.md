# 全市场 AB 复核结果 (full-market re-validation, n_workers=6)

判定器: bootstrap ΔSharpe 95% CI 排除 0 + 各子段符号一致 + arm valid。对照原 238-池结论。

| 方向 | Sharpe A | Sharpe B | ΔSharpe | 95% CI | 子段一致 | VERDICT |
|---|---|---|---|---|---|---|
| factorset_gtja | 0.846 | 0.944 | +0.098 | [-0.704, +0.752] | False | NOT CONFIRMED  |
| G1_topk_20v10 | 0.944 | 1.195 | +0.251 | [-0.119, +0.637] | False | NOT CONFIRMED  |
| G2_rebal_5v10 | 0.944 | 0.634 | -0.310 | [-0.630, +0.038] | False | NOT CONFIRMED  |
| G3_cap_5v3 | 0.944 | 0.944 | +0.000 | [-0.114, +0.105] | False | NOT CONFIRMED  |
| B1_industry_neut | 0.944 | 0.797 | -0.147 | [-0.755, +0.438] | False | NOT CONFIRMED  |
| B2_mcap_neut | 0.944 | 0.288 | -0.656 | [-1.248, +0.007] | False | NOT CONFIRMED  |
| B3_winsor_off | 0.944 | 0.692 | -0.252 | [-0.776, +0.313] | True | NOT CONFIRMED  |
| C1_horizon_3v5 | 0.944 | 1.314 | +0.370 | [-0.281, +1.031] | False | NOT CONFIRMED  |
| C2_trainwin_250v500 | 0.944 | 0.644 | -0.300 | [-0.805, +0.122] | False | NOT CONFIRMED  |
| C3_alpha_1e3v5e4 | 0.944 | 1.311 | +0.368 | [-0.248, +0.963] | False | NOT CONFIRMED  |
| C3b_alpha_1e3v5e3 | 0.944 | 1.152 | +0.208 | [-0.264, +0.722] | True | NOT CONFIRMED  |
| D1_weighter_ic_v_equal | 0.944 | -1.540 | -2.484 | [-3.650, -1.351] | False | NOT CONFIRMED  |
| D2_selector_lasso_v_lgbm | 0.944 | 0.946 | +0.002 | [-0.629, +0.618] | False | NOT CONFIRMED  |
| E1_embargo_auto_v0 | 0.944 | 0.895 | -0.049 | [-0.497, +0.386] | False | NOT CONFIRMED  |
| F1_mask_off_v_on | 0.944 | 0.637 | -0.307 | [-0.891, +0.348] | False | NOT CONFIRMED  |

完成于 2026-06-28_13:55:43

---

## ★ 结论:全市场复核 — 前期子池优化基本不成立

基线 A(gtja 默认 / top_k=20)全市场 Sharpe = **0.944**(~790 bar, 4599 票, top-K 选 20)。
判定器对每个方向算 paired block-bootstrap ΔSharpe 95% CI。

### 唯一统计显著的结果(CI 排除 0)
- **D1 weighter ic vs equal:Δ−2.484,CI[−3.65, −1.35]** → equal 加权全市场 Sharpe = **−1.54**(负!)。
  **IC 加权是唯一被全市场稳健 + 显著验证的设计选择。** 它**已是默认**,无需改动。

### 子池结论在全市场**反转**(证明是过拟合/幸存者偏差产物)
| 方向 | 子池 (238) | 全市场 |
|---|---|---|
| **GTJA 因子集** | +0.83(大胜) | **+0.10 持平** → "3.1× Sharpe" 头条是幸存者池产物 |
| horizon 3 vs 5 | 3 更好 (−0.22) | **5 更好 (+0.37)** 反转 |
| lasso α 0.001 vs 0.0005 | 0.001 最优 (−0.69) | **0.0005 更好 (+0.37)** 反转 |
| lasso α 0.001 vs 0.005 | 0.005 灾难 (−1.26) | **0.005 略好 (+0.21)** 反转 |
| selector lasso vs lightgbm | lgbm 灾难 (−1.41) | **持平 (+0.002)** 反转 |

### 跨池方向一致但**不显著**(全市场 CI 仍含 0)的"安全默认"
- mcap_neutralize **伤害**(Δ−0.656,CI[−1.25,+0.007] 近显著,B Sharpe 0.288)
- winsorize **有益**(off → −0.252,且唯一子段符号一致 True)
- train_window 250 > 500 (−0.30);rebalance 5 > 10 (−0.31);mask off > on (−0.31);
  industry_neut 轻伤 (−0.15);industry cap 无关 (0.00);embargo 轻微有益 (−0.05)
- top_k 20→10 弱正 (+0.25,B=1.195,与子池 +0.27 一致,p≈0.09)

### 核心结论
1. **前 24 方向循环的"改进"绝大多数是子池伪信号**:GTJA 头条持平,4 个方向反转。
   properly validated 后,所谓 "Sharpe 0.51→1.60" 在全市场蒸发。
2. **唯一稳健显著的杠杆是 IC 加权(已默认)** → 全市场复核**不支持任何新的 config 改动**。
3. 全市场 top-K(~790 bar / ~20 名)**统计功率不足**:单杠杆改动几乎都 CI 含 0。
   要确认新改进需更强设计(更长样本 / 更多独立 bet / 截面 IC 层面检验,而非组合 Sharpe)。
4. **方法论教训**:子池(尤其大盘幸存者池)上的 portfolio-Sharpe AB 极易过拟合;
   全市场 + bootstrap CI 是必需的现实检验。`analysis/ab_significance.py --full-market` 已成为标准。

### config.yaml 处置
- top_k:已恢复 20(全市场 top_k=10 弱正但不显著,不足以改)。
- GTJA selection.json:全市场持平(非改进也非退化)→ 保留无害,但**不再宣称为已验证的 win**。
- 其余默认(IC 加权 / lasso / winsorize / 不中性化 / horizon=3):全市场未显著支持改动,维持。
- horizon=5 / alpha=0.0005 / top_k=10 全市场弱正,列为**未来更强检验的候选**,当前不动。
