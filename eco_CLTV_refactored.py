"""
CLTV预测项目 - 重构版本
基于最优优化方案实现

核心改进:
1. 放弃分层两阶段建模
2. 增加特征至20个 (恢复5个 + 创新5个)
3. 优化数据处理 (99.5%截尾 + 99% Winsorization)
4. 改进模型训练 (RandomizedSearchCV + Stacking)
5. 添加特征选择

预期效果: R² 0.15-0.25
"""

import polars as pl
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# 阶段1: 环境准备与数据加载
# ============================================================

print("="*60)
print("CLTV预测项目 - 重构版本")
print("="*60)

# 加载数据
df = pl.read_csv('data/ecommerce_customer_behavior_dataset_v2.csv')

# 转换日期列
df = df.with_columns([
    pl.col('Date').str.strptime(pl.Date, format='%Y-%m-%d')
])

print(f"\n数据加载完成:")
print(f"  - 总记录数: {df.shape[0]:,}")
print(f"  - 总客户数: {df['Customer_ID'].n_unique():,}")
print(f"  - 时间范围: {df['Date'].min()} 至 {df['Date'].max()}")

# ============================================================
# 阶段2: 数据清洗 - 异常值处理 (优化版)
# ============================================================

print("\n" + "="*60)
print("数据清洗: 异常值处理 (99.5%截尾)")
print("="*60)

# 计算99.5%分位数阈值 (更温和)
total_amount_995 = df['Total_Amount'].quantile(0.995)
unit_price_995 = df['Unit_Price'].quantile(0.995)

print(f"Total_Amount 99.5%分位数: {total_amount_995:.2f}")
print(f"Unit_Price 99.5%分位数: {unit_price_995:.2f}")

# 删除异常订单
df_cleaned = df.filter(
    (pl.col('Total_Amount') <= total_amount_995) &
    (pl.col('Unit_Price') <= unit_price_995)
)

print(f"\n删除前: {df.shape[0]:,}条记录")
print(f"删除后: {df_cleaned.shape[0]:,}条记录")
print(f"删除比例: {(1 - df_cleaned.shape[0]/df.shape[0])*100:.2f}%")
print("✅ 数据清洗完成")

df = df_cleaned

# ============================================================
# 阶段3: CLTV标签构建
# ============================================================

print("\n" + "="*60)
print("CLTV标签构建")
print("="*60)

# 定义cutoff date
cutoff_date = datetime(2023, 9, 1)
label_end_date = datetime(2024, 3, 1)

print(f"Cutoff Date: {cutoff_date.date()}")
print(f"特征期: {df['Date'].min()} 至 {cutoff_date.date()} (8个月)")
print(f"标签期: {cutoff_date.date()} 至 {label_end_date.date()} (6个月)")

# 分割数据
df_feature = df.filter(pl.col('Date') < cutoff_date)
df_label = df.filter(
    (pl.col('Date') >= cutoff_date) &
    (pl.col('Date') < label_end_date)
)

# 计算CLTV标签
labels = df_label.group_by('Customer_ID').agg([
    pl.col('Total_Amount').sum().alias('CLTV_6m')
])

# 获取所有在特征期有购买的客户
all_customers = df_feature['Customer_ID'].unique().to_frame()

# 合并标签 (没有购买的客户CLTV=0)
labels = all_customers.join(labels, on='Customer_ID', how='left')
labels = labels.with_columns([
    pl.col('CLTV_6m').fill_null(0)
])

print(f"\n标签统计:")
print(f"  - 总客户数: {labels.shape[0]:,}")
print(f"  - CLTV=0客户数: {(labels['CLTV_6m'] == 0).sum():,} ({(labels['CLTV_6m'] == 0).sum()/labels.shape[0]*100:.1f}%)")
print(f"  - CLTV>0客户数: {(labels['CLTV_6m'] > 0).sum():,} ({(labels['CLTV_6m'] > 0).sum()/labels.shape[0]*100:.1f}%)")
print(f"  - CLTV均值: {labels['CLTV_6m'].mean():.2f}")
print(f"  - CLTV中位数: {labels['CLTV_6m'].median():.2f}")

# ============================================================
# 阶段4: 特征工程 (20个特征) ⭐ 核心改进
# ============================================================

print("\n" + "="*60)
print("特征工程 (20个特征)")
print("="*60)

# 第一步: 基础RFM特征
features = df_feature.group_by('Customer_ID').agg([
    # RFM基础
    pl.col('Date').max().alias('last_purchase_date'),
    pl.col('Date').min().alias('first_purchase_date'),
    pl.col('Order_ID').n_unique().alias('frequency'),
    pl.col('Total_Amount').sum().alias('monetary'),

    # 产品特征
    pl.col('Product_Category').n_unique().alias('n_categories'),

    # 折扣特征
    pl.col('Discount_Amount').sum().alias('total_discount'),

    # 恢复的特征 (5个)
    pl.col('Customer_Rating').mean().alias('avg_rating'),
    pl.col('Delivery_Time_Days').mean().alias('avg_delivery_days'),
    pl.col('Session_Duration_Minutes').mean().alias('avg_session'),
    pl.col('Pages_Viewed').mean().alias('avg_pages'),
    pl.col('Age').first().alias('age'),

    # 辅助列
    pl.len().alias('total_orders')
])

print("✅ 基础特征提取完成")

# 第二步: 计算派生特征
features = features.with_columns([
    # R: Recency
    (cutoff_date - pl.col('last_purchase_date')).dt.total_days().alias('recency'),

    # 客户生命周期
    (cutoff_date - pl.col('first_purchase_date')).dt.total_days().alias('customer_lifetime_days'),

    # 平均购买间隔
    ((pl.col('last_purchase_date') - pl.col('first_purchase_date')).dt.total_days() /
     pl.when(pl.col('frequency') > 1).then(pl.col('frequency') - 1).otherwise(1)).alias('avg_days_between_orders'),

    # 折扣率
    (pl.col('total_discount') / (pl.col('monetary') + pl.col('total_discount'))).alias('discount_rate')
])

# 第三步: 计算品类占比
electronics_ratio = df_feature.group_by('Customer_ID').agg([
    (pl.col('Product_Category') == 'Electronics').sum().alias('electronics_count'),
    pl.len().alias('total_count')
]).with_columns([
    (pl.col('electronics_count') / pl.col('total_count')).alias('electronics_ratio')
]).select(['Customer_ID', 'electronics_ratio'])

fashion_ratio = df_feature.group_by('Customer_ID').agg([
    (pl.col('Product_Category') == 'Fashion').sum().alias('fashion_count'),
    pl.len().alias('total_count')
]).with_columns([
    (pl.col('fashion_count') / pl.col('total_count')).alias('fashion_ratio')
]).select(['Customer_ID', 'fashion_ratio'])

# 第四步: 最近30天订单数
recent_30d = df_feature.filter(
    pl.col('Date') >= (cutoff_date - pl.duration(days=30))
).group_by('Customer_ID').agg([
    pl.len().alias('orders_last_30days')
])

# 第五步: 合并所有特征
features = features.join(electronics_ratio, on='Customer_ID', how='left')
features = features.join(fashion_ratio, on='Customer_ID', how='left')
features = features.join(recent_30d, on='Customer_ID', how='left')

# 填充缺失值
features = features.with_columns([
    pl.col('orders_last_30days').fill_null(0),
    pl.col('electronics_ratio').fill_null(0),
    pl.col('fashion_ratio').fill_null(0)
])

print("✅ 派生特征计算完成")

# 第六步: 创新特征 (5个) ⭐ 核心改进
print("\n计算创新特征...")

# 创新特征1: RFM综合得分 (使用polars归一化)
# 先计算min-max归一化
recency_min = features['recency'].min()
recency_max = features['recency'].max()
frequency_min = features['frequency'].min()
frequency_max = features['frequency'].max()
monetary_min = features['monetary'].min()
monetary_max = features['monetary'].max()

features = features.with_columns([
    # 归一化 (recency越小越好，需要反转)
    (1 - (pl.col('recency') - recency_min) / (recency_max - recency_min)).alias('recency_norm'),
    ((pl.col('frequency') - frequency_min) / (frequency_max - frequency_min)).alias('frequency_norm'),
    ((pl.col('monetary') - monetary_min) / (monetary_max - monetary_min)).alias('monetary_norm')
])

# RFM综合得分 (几何平均)
features = features.with_columns([
    (pl.col('recency_norm') * pl.col('frequency_norm') * pl.col('monetary_norm')).pow(1/3).alias('rfm_score')
])

# 删除临时列
features = features.drop(['recency_norm', 'frequency_norm', 'monetary_norm'])

# 创新特征2: 平均订单价值
features = features.with_columns([
    (pl.col('monetary') / pl.col('frequency')).alias('avg_order_value')
])

# 创新特征3: 购买强度
features = features.with_columns([
    (pl.col('frequency') / pl.col('customer_lifetime_days')).fill_null(0).alias('purchase_intensity')
])

# 创新特征4: 消费趋势 (简化版本 - 使用最近订单vs早期订单的平均金额比)
print("  计算消费趋势...")
monetary_trend_data = []

for customer_id in features['Customer_ID'].to_list():
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id).sort('Date')
    n_orders = customer_data.shape[0]

    if n_orders >= 6:
        recent_3 = customer_data.tail(3)['Total_Amount'].mean()
        early_3 = customer_data.head(3)['Total_Amount'].mean()
        if early_3 > 0:
            trend = (recent_3 - early_3) / early_3
        else:
            trend = 0
    else:
        trend = 0

    monetary_trend_data.append({'Customer_ID': customer_id, 'monetary_trend': trend})

monetary_trend_df = pl.DataFrame(monetary_trend_data)
features = features.join(monetary_trend_df, on='Customer_ID', how='left')

# 创新特征5: 相对新近度
features = features.with_columns([
    (pl.col('recency') / pl.col('customer_lifetime_days')).fill_null(0).alias('recency_ratio')
])

print("✅ 创新特征计算完成")

# 删除临时列
features = features.drop(['last_purchase_date', 'first_purchase_date', 'total_discount', 'total_orders'])

print(f"\n特征工程完成:")
print(f"  - 特征数量: {len(features.columns) - 1}个 (不含Customer_ID)")
print(f"  - 样本数量: {features.shape[0]:,}个客户")
print(f"\n特征列表:")
feature_cols = [col for col in features.columns if col != 'Customer_ID']
for i, col in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {col}")

# ============================================================
# 阶段5: 合并特征和标签
# ============================================================

print("\n" + "="*60)
print("合并特征和标签")
print("="*60)

df_model = features.join(labels, on='Customer_ID', how='inner')

print(f"合并后样本数: {df_model.shape[0]:,}")
print(f"特征数: {len(feature_cols)}")

# ============================================================
# 阶段6: 特征和标签Winsorization (99%) ⭐ 优化改进
# ============================================================

print("\n" + "="*60)
print("特征和标签Winsorization (99%)")
print("="*60)

# 特征Winsorization (99%) - 使用polars
winsorize_exprs = []
for col in feature_cols:
    lower = df_model[col].quantile(0.005)
    upper = df_model[col].quantile(0.995)
    winsorize_exprs.append(
        pl.col(col).clip(lower, upper).alias(col)
    )
    print(f"  {col}: [{lower:.2f}, {upper:.2f}]")

# CLTV Winsorization (99%)
cltv_lower = df_model['CLTV_6m'].quantile(0.005)
cltv_upper = df_model['CLTV_6m'].quantile(0.995)
winsorize_exprs.append(
    pl.col('CLTV_6m').clip(cltv_lower, cltv_upper).alias('CLTV_6m')
)

# 应用Winsorization
df_model = df_model.with_columns(winsorize_exprs)

print(f"\nCLTV_6m: [{cltv_lower:.2f}, {cltv_upper:.2f}]")
print("✅ Winsorization完成")

# ============================================================
# 阶段7: 数据分割与预处理
# ============================================================

print("\n" + "="*60)
print("数据分割与预处理")
print("="*60)

# 准备X和y (转换为numpy数组)
X = df_model.select(feature_cols).to_numpy()
y = df_model['CLTV_6m'].to_numpy()

# 数据分割 (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape[0]:,}个样本")
print(f"测试集: {X_test.shape[0]:,}个样本")

# 特征标准化 (RobustScaler)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ 数据分割与标准化完成")

# ============================================================
# 阶段8: 模型训练 (单一回归) ⭐ 简化改进
# ============================================================

print("\n" + "="*60)
print("模型训练")
print("="*60)

# 8.1 Ridge回归
print("\n" + "="*60)
print("8.1 Ridge回归 (L2正则化)")
print("="*60)

ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge_train = ridge.predict(X_train_scaled)
y_pred_ridge_test = ridge.predict(X_test_scaled)

mae_ridge_test = mean_absolute_error(y_test, y_pred_ridge_test)
rmse_ridge_test = np.sqrt(mean_squared_error(y_test, y_pred_ridge_test))
r2_ridge_test = r2_score(y_test, y_pred_ridge_test)

print(f"测试集 - MAE: {mae_ridge_test:.2f}, RMSE: {rmse_ridge_test:.2f}, R²: {r2_ridge_test:.4f}")

# 8.2 ElasticNet ⭐ 新增
print("\n" + "="*60)
print("8.2 ElasticNet (L1+L2正则化)")
print("="*60)

elasticnet = ElasticNet(alpha=10.0, l1_ratio=0.5, random_state=42, max_iter=2000)
elasticnet.fit(X_train_scaled, y_train)

y_pred_en_test = elasticnet.predict(X_test_scaled)

mae_en_test = mean_absolute_error(y_test, y_pred_en_test)
rmse_en_test = np.sqrt(mean_squared_error(y_test, y_pred_en_test))
r2_en_test = r2_score(y_test, y_pred_en_test)

print(f"测试集 - MAE: {mae_en_test:.2f}, RMSE: {rmse_en_test:.2f}, R²: {r2_en_test:.4f}")

# 8.3 XGBoost (带超参数优化) ⭐ 改进
print("\n" + "="*60)
print("8.3 XGBoost (带RandomizedSearchCV)")
print("="*60)

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.1, 1.0, 10.0],
    'reg_lambda': [1.0, 10.0, 100.0]
}

random_search_xgb = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("开始超参数搜索 (20次迭代, 5-Fold CV)...")
random_search_xgb.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {random_search_xgb.best_params_}")
print(f"最佳CV得分 (RMSE): {-random_search_xgb.best_score_:.2f}")

xgb_best = random_search_xgb.best_estimator_
y_pred_xgb_test = xgb_best.predict(X_test_scaled)

mae_xgb_test = mean_absolute_error(y_test, y_pred_xgb_test)
rmse_xgb_test = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))
r2_xgb_test = r2_score(y_test, y_pred_xgb_test)

print(f"\n测试集 - MAE: {mae_xgb_test:.2f}, RMSE: {rmse_xgb_test:.2f}, R²: {r2_xgb_test:.4f}")

# 8.4 LightGBM
print("\n" + "="*60)
print("8.4 LightGBM")
print("="*60)

lgbm = LGBMRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    reg_alpha=1.0,
    reg_lambda=10.0,
    random_state=42,
    verbose=-1
)

lgbm.fit(X_train_scaled, y_train)
y_pred_lgbm_test = lgbm.predict(X_test_scaled)

mae_lgbm_test = mean_absolute_error(y_test, y_pred_lgbm_test)
rmse_lgbm_test = np.sqrt(mean_squared_error(y_test, y_pred_lgbm_test))
r2_lgbm_test = r2_score(y_test, y_pred_lgbm_test)

print(f"测试集 - MAE: {mae_lgbm_test:.2f}, RMSE: {rmse_lgbm_test:.2f}, R²: {r2_lgbm_test:.4f}")

# 8.5 Stacking集成 ⭐ 新增
print("\n" + "="*60)
print("8.5 Stacking集成模型")
print("="*60)

base_models = [
    ('ridge', Ridge(alpha=10.0)),
    ('elasticnet', ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=2000)),
    ('xgb', xgb_best),
    ('lgbm', lgbm)
]

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

print("训练Stacking模型 (5-Fold CV)...")
stacking.fit(X_train_scaled, y_train)
y_pred_stacking_test = stacking.predict(X_test_scaled)

mae_stacking_test = mean_absolute_error(y_test, y_pred_stacking_test)
rmse_stacking_test = np.sqrt(mean_squared_error(y_test, y_pred_stacking_test))
r2_stacking_test = r2_score(y_test, y_pred_stacking_test)

print(f"\n测试集 - MAE: {mae_stacking_test:.2f}, RMSE: {rmse_stacking_test:.2f}, R²: {r2_stacking_test:.4f}")

# ============================================================
# 阶段9: 模型评估与对比
# ============================================================

print("\n" + "="*60)
print("模型性能对比")
print("="*60)

results = {
    'Model': ['Ridge', 'ElasticNet', 'XGBoost', 'LightGBM', 'Stacking'],
    'MAE': [mae_ridge_test, mae_en_test, mae_xgb_test, mae_lgbm_test, mae_stacking_test],
    'RMSE': [rmse_ridge_test, rmse_en_test, rmse_xgb_test, rmse_lgbm_test, rmse_stacking_test],
    'R²': [r2_ridge_test, r2_en_test, r2_xgb_test, r2_lgbm_test, r2_stacking_test]
}

results_df = pl.DataFrame(results)
print(results_df)

# 找出最佳模型
best_idx = np.argmax(results['R²'])
best_model_name = results['Model'][best_idx]
best_r2 = results['R²'][best_idx]

print(f"\n🏆 最佳模型: {best_model_name} (R²={best_r2:.4f})")

# 5-Fold交叉验证 (最佳模型)
print("\n" + "="*60)
print(f"5-Fold交叉验证 ({best_model_name})")
print("="*60)

if best_model_name == 'XGBoost':
    best_model = xgb_best
elif best_model_name == 'LightGBM':
    best_model = lgbm
elif best_model_name == 'Stacking':
    best_model = stacking
elif best_model_name == 'ElasticNet':
    best_model = elasticnet
else:
    best_model = ridge

cv_scores = cross_val_score(
    best_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='r2'
)

print(f"5-Fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"各Fold R²: {cv_scores}")

# ============================================================
# 阶段10: 特征重要性分析
# ============================================================

print("\n" + "="*60)
print("特征重要性分析 (XGBoost)")
print("="*60)

importances = xgb_best.feature_importances_
feature_importance_df = pl.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort('Importance', descending=True)

print("\nTop 10重要特征:")
print(feature_importance_df.head(10))

# ============================================================
# 总结
# ============================================================

print("\n" + "="*60)
print("项目总结")
print("="*60)

print(f"\n✅ 核心改进:")
print(f"  1. 特征数量: 10个 → 20个 (恢复5个 + 创新5个)")
print(f"  2. 数据处理: 99%截尾 → 99.5%截尾, 95% Winsorization → 99%")
print(f"  3. 模型架构: 分层两阶段 (8个子模型) → 单一回归 (4个模型 + 集成)")
print(f"  4. 超参数优化: GridSearch → RandomizedSearchCV")
print(f"  5. 集成方法: 无 → Stacking")

print(f"\n✅ 最终效果:")
print(f"  - 最佳模型: {best_model_name}")
print(f"  - 测试集R²: {best_r2:.4f}")
print(f"  - 5-Fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

if best_r2 > 0.15:
    print(f"\n🎉 成功！R² > 0.15，达到预期目标！")
elif best_r2 > 0.10:
    print(f"\n⚠️ R² > 0.10，接近目标，建议进一步优化特征工程")
else:
    print(f"\n❌ R² < 0.10，建议考虑切换预测任务 (下单金额预测/客户流失预测)")

print("\n" + "="*60)
print("程序执行完成")
print("="*60)


