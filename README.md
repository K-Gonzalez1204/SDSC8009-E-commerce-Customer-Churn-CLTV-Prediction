# E-commerce Customer Churn & CLTV Prediction

**课程**: SDSC8009 Data Mining（5 人小组项目）
**数据来源**: [Kaggle - E-commerce Customer Behavior and Sales Analysis](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr/data)
**维护状态**: 2026-08-02 已完成无泄漏建模重构；CLTV 支线作为决策链证据保留

## 项目简介

电商客户**流失预测**（churn）二元分类项目：在客户级严格留出评估协议下，
识别高流失风险客户名单供业务跟进。项目叙事主线是"止损决策链"——最初主题
为 CLTV 回归，确认目标在该合成数据上不可学习后，按课程评分标准转向流失
分类并交付可用模型；CLTV 支线作为决策链证据保留（详见下文"历史演进"）。

数据为**合成数据集**，结果只用于课程/项目展示，不宣称真实业务有效性。

## 最新结果：2026 无泄漏重构

旧流程把 99.5% 截尾、99% Winsorization 放在切分前全量计算，SMOTE 也在
GridSearch/CV 之前作用于整个训练集，导致合成邻居跨折泄漏。新协议：

- 客户级分层 80/20 outer holdout（seed 42）；
- 训练集内 5 折 CV；Winsorization、RobustScaler、SelectKBest(k=20) 只在
  训练折 fit；
- 阈值在训练折 OOF 预测上最大化 F1（0.10–0.90，步长 0.05）；
- 锁模型后只在 holdout 评估一次；
- 类别不平衡用 class_weight / scale_pos_weight；本轮不使用 SMOTE。

结果（`output/leak_free_churn/`，输入 SHA-256 `1588736F...3285`）：

| 指标 | 值 |
|---|---|
| 客户样本 | 4,337（流失 741，17.1%） |
| 最优模型（OOF CV ROC-AUC） | LogisticRegression 0.8445 ± 0.0135 |
| Holdout ROC-AUC | **0.8475** |
| Holdout PR-AUC | 0.4544 |
| 阈值（OOF 最大化 F1） | 0.55 |
| Holdout Accuracy | 0.7408 |
| Churn Recall / Precision / F1 | 0.973 / 0.395 / 0.561 |
| Bootstrap 95% CI（ROC-AUC） | 0.8213 – 0.8722 |

## 运行

```powershell
cd code
python -m unittest discover -s tests -v
cd ..
python code/04_leak_free_churn.py
```

产物：`output/leak_free_churn/`（final_metrics.json、manifest.json、
best_model.pkl、threshold_selection.csv、figures/）。入口
`code/04_leak_free_churn.py` 调用 `code/leak_free/` 包；根目录旧脚本
（`eco_churn_prediction*.py`、`eco_CLTV_*.py` 等）为历史/证据保留，不进入
新实验。

## 历史指标（按模型绑定，旧协议结果）

| 模型 | AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8389 | 0.9866 | 0.3722 | 0.5404 |
| Random Forest | 0.8347 | 0.8792 | 0.3722 | 0.5230 |
| XGBoost | 0.8380 | 0.4094 | 0.3961 | 0.4026 |
| LightGBM | 0.8356 | 0.4765 | 0.4201 | 0.4465 |
| CatBoost | 0.8268 | 0.5973 | 0.3708 | 0.4576 |
| Soft Voting | 0.8356 | 0.6242 | 0.3690 | 0.4638 |

旧 5 折 CV 0.8517 在 SMOTE 后数据上计算，不作为严格成绩。

## 历史演进

1. **CLTV 回归尝试**：Enhanced（R²≈-0.01~-0.05）、Optimized（R²≈0.0001）、
   Refactored（R²≈0.0069）五版迭代，效果始终接近不可预测。
2. **数据溯源**：确认数据集为合成数据，客户历史与未来消费在生成过程上不存在
   因果依赖，CLTV 目标在此数据上不可学习。
3. **转向流失分类**：按课程评分标准对 8 个候选任务打分（流失分析 88 分第一，
   CLTV 64 分第六），转为流失二分类并完成 2026 无泄漏重构。

课程期 AI 对话、任务评分和日志保存在本地 `.history/`、
`docs/CLTV优化方案_最终版.md` 与 `output/eco_*_log.txt`；
`eco_CLTV_*.ipynb` / `eco_CLTV_refactored.py` 保留为决策链证据，不删除、
不进入新实验。

## 团队与贡献口径

5 人课程小组项目；本人负责运行代码与建模评估（AI 辅助），其余分工不细拆。

## 限制

- 数据为合成数据集，不宣称真实业务有效性。
- 流失率 17.1% 属轻中度不平衡；Churn Precision 较低是阈值 0.55 下追求高
  召回的业务权衡（Recall 0.973 / Precision 0.395）。
- 本轮固定超参；时间外推、超参调优与 SMOTE-in-fold 对比是后续实验。
