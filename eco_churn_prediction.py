"""
============================================================
电商客户流失预测项目 (E-commerce Customer Churn Prediction)
============================================================

项目目标: 预测未来6个月客户是否流失
预期效果: AUC 0.75-0.85, Accuracy 0.72-0.80

技术栈:
- Polars: 数据处理
- Scikit-learn: 机器学习
- XGBoost, LightGBM: 梯度提升
- Lets-Plot: 可视化

作者: AI Assistant
日期: 2025-11-18
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('eco_churn_prediction_log.txt', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(message):
    """统一的日志输出函数"""
    logging.info(message)

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

log("=" * 60)
log("电商客户流失预测项目")
log("=" * 60)
log(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")

# ============================================================
# 阶段1: 数据加载
# ============================================================
log("=" * 60)
log("阶段1: 数据加载")
log("=" * 60)

df = pl.read_csv('data/ecommerce_customer_behavior_dataset_v2.csv')
df = df.with_columns(pl.col('Date').str.to_date('%Y-%m-%d'))

log(f"数据加载完成:")
log(f"  - 总记录数: {len(df):,}")
log(f"  - 总客户数: {df['Customer_ID'].n_unique():,}")
log(f"  - 时间范围: {df['Date'].min()} 至 {df['Date'].max()}")
log("")

# ============================================================
# 阶段2: 数据清洗 (99.5%截尾)
# ============================================================
log("=" * 60)
log("阶段2: 数据清洗 (异常值处理)")
log("=" * 60)

# 计算99.5%分位数
total_amount_995 = df['Total_Amount'].quantile(0.995)
unit_price_995 = df['Unit_Price'].quantile(0.995)

log(f"Total_Amount 99.5%分位数: {total_amount_995:.2f}")
log(f"Unit_Price 99.5%分位数: {unit_price_995:.2f}")
log("")

# 删除异常订单
before_count = len(df)
df_cleaned = df.filter(
    (pl.col('Total_Amount') <= total_amount_995) &
    (pl.col('Unit_Price') <= unit_price_995)
)
after_count = len(df_cleaned)

log(f"删除前: {before_count:,}条记录")
log(f"删除后: {after_count:,}条记录")
log(f"删除比例: {(before_count - after_count) / before_count * 100:.2f}%")
log("✅ 数据清洗完成")
log("")

# ============================================================
# 阶段3: 流失标签构建
# ============================================================
log("=" * 60)
log("阶段3: 流失标签构建")
log("=" * 60)

# 定义cutoff date
cutoff_date = datetime(2023, 9, 1).date()
feature_end_date = cutoff_date
label_end_date = datetime(2024, 3, 1).date()

log(f"Cutoff Date: {cutoff_date}")
log(f"特征期: {df_cleaned['Date'].min()} 至 {feature_end_date} (8个月)")
log(f"标签期: {cutoff_date} 至 {label_end_date} (6个月)")
log("")

# 特征期数据
df_feature = df_cleaned.filter(pl.col('Date') < cutoff_date)

# 标签期数据
df_label = df_cleaned.filter(
    (pl.col('Date') >= cutoff_date) & 
    (pl.col('Date') < label_end_date)
)

# 构建流失标签: 标签期有购买=0(未流失), 无购买=1(流失)
label_customers = df_label.select('Customer_ID').unique()
feature_customers = df_feature.select('Customer_ID').unique()

# 所有在特征期有购买的客户
all_customers = feature_customers.with_columns(
    pl.lit(1).alias('is_churn')  # 默认流失
)

# 标签期有购买的客户标记为未流失
all_customers = all_customers.join(
    label_customers.with_columns(pl.lit(0).alias('is_churn_new')),
    on='Customer_ID',
    how='left'
).with_columns(
    pl.when(pl.col('is_churn_new').is_not_null())
    .then(0)
    .otherwise(1)
    .alias('is_churn')
).select(['Customer_ID', 'is_churn'])

log(f"标签统计:")
log(f"  - 总客户数: {len(all_customers):,}")
log(f"  - 流失客户数: {all_customers.filter(pl.col('is_churn') == 1).height:,} ({all_customers.filter(pl.col('is_churn') == 1).height / len(all_customers) * 100:.1f}%)")
log(f"  - 未流失客户数: {all_customers.filter(pl.col('is_churn') == 0).height:,} ({all_customers.filter(pl.col('is_churn') == 0).height / len(all_customers) * 100:.1f}%)")
log("")

# ============================================================
# 阶段4: 特征工程 (25个特征)
# ============================================================
log("=" * 60)
log("阶段4: 特征工程 (25个特征)")
log("=" * 60)

# 4.1 基础RFM特征 (5个)
customer_ids = []
recency_list = []
frequency_list = []
monetary_list = []
customer_lifetime_days_list = []
avg_days_between_orders_list = []

for customer_id in df_feature['Customer_ID'].unique().sort():
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)

    # Recency: 距离cutoff date的天数
    last_purchase_date = customer_data['Date'].max()
    recency = (cutoff_date - last_purchase_date).days

    # Frequency: 购买次数
    frequency = len(customer_data)

    # Monetary: 总消费金额
    monetary = customer_data['Total_Amount'].sum()

    # Customer Lifetime: 客户生命周期天数
    first_purchase_date = customer_data['Date'].min()
    customer_lifetime_days = (last_purchase_date - first_purchase_date).days + 1

    # Avg Days Between Orders: 平均订单间隔天数
    if frequency > 1:
        avg_days_between_orders = customer_lifetime_days / (frequency - 1)
    else:
        avg_days_between_orders = 0

    customer_ids.append(customer_id)
    recency_list.append(recency)
    frequency_list.append(frequency)
    monetary_list.append(monetary)
    customer_lifetime_days_list.append(customer_lifetime_days)
    avg_days_between_orders_list.append(avg_days_between_orders)

# 4.2 行为特征 (7个)
n_categories_list = []
avg_rating_list = []
avg_delivery_days_list = []
avg_session_list = []
avg_pages_list = []
discount_rate_list = []
orders_last_30days_list = []

for customer_id in df_feature['Customer_ID'].unique().sort():
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)

    # 购买品类数
    n_categories = customer_data['Product_Category'].n_unique()

    # 平均评分
    avg_rating = customer_data['Customer_Rating'].mean()

    # 平均配送天数
    avg_delivery_days = customer_data['Delivery_Time_Days'].mean()

    # 平均会话时长
    avg_session = customer_data['Session_Duration_Minutes'].mean()

    # 平均浏览页面数
    avg_pages = customer_data['Pages_Viewed'].mean()

    # 折扣率
    total_discount = customer_data['Discount_Amount'].sum()
    total_amount = customer_data['Total_Amount'].sum()
    discount_rate = total_discount / (total_amount + total_discount) if (total_amount + total_discount) > 0 else 0

    # 最近30天订单数
    last_purchase_date = customer_data['Date'].max()
    date_30days_ago = last_purchase_date - timedelta(days=30)
    orders_last_30days = customer_data.filter(pl.col('Date') >= date_30days_ago).height

    n_categories_list.append(n_categories)
    avg_rating_list.append(avg_rating)
    avg_delivery_days_list.append(avg_delivery_days)
    avg_session_list.append(avg_session)
    avg_pages_list.append(avg_pages)
    discount_rate_list.append(discount_rate)
    orders_last_30days_list.append(orders_last_30days)

# 4.3 产品偏好特征 (4个)
electronics_ratio_list = []
fashion_ratio_list = []
home_ratio_list = []
beauty_ratio_list = []

for customer_id in df_feature['Customer_ID'].unique().sort():
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)
    total_orders = len(customer_data)

    electronics_ratio = customer_data.filter(pl.col('Product_Category') == 'Electronics').height / total_orders
    fashion_ratio = customer_data.filter(pl.col('Product_Category') == 'Fashion').height / total_orders
    home_ratio = customer_data.filter(pl.col('Product_Category') == 'Home & Kitchen').height / total_orders
    beauty_ratio = customer_data.filter(pl.col('Product_Category') == 'Beauty & Personal Care').height / total_orders

    electronics_ratio_list.append(electronics_ratio)
    fashion_ratio_list.append(fashion_ratio)
    home_ratio_list.append(home_ratio)
    beauty_ratio_list.append(beauty_ratio)

# 4.4 支付与设备特征 (3个)
mobile_ratio_list = []
credit_card_ratio_list = []
is_returning_ratio_list = []

for customer_id in df_feature['Customer_ID'].unique().sort():
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)
    total_orders = len(customer_data)

    mobile_ratio = customer_data.filter(pl.col('Device_Type') == 'Mobile').height / total_orders
    credit_card_ratio = customer_data.filter(pl.col('Payment_Method') == 'Credit Card').height / total_orders
    is_returning_ratio = customer_data['Is_Returning_Customer'].mean()

    mobile_ratio_list.append(mobile_ratio)
    credit_card_ratio_list.append(credit_card_ratio)
    is_returning_ratio_list.append(is_returning_ratio)

# 4.5 人口统计特征 (1个)
age_list = []

for customer_id in df_feature['Customer_ID'].unique().sort():
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)
    age = customer_data['Age'].mode()[0] if len(customer_data['Age'].mode()) > 0 else customer_data['Age'].mean()
    age_list.append(int(age))  # 转换为整数

# 4.6 创新特征 (5个)
log("计算创新特征...")

rfm_score_list = []
avg_order_value_list = []
purchase_intensity_list = []
monetary_trend_list = []
recency_ratio_list = []

for i, customer_id in enumerate(df_feature['Customer_ID'].unique().sort()):
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)

    # RFM综合评分 (归一化后的加权和)
    r_norm = 1 - (recency_list[i] / 243)  # 越小越好，所以取反
    f_norm = frequency_list[i] / 10  # 越大越好
    m_norm = monetary_list[i] / 20000  # 越大越好
    rfm_score = 0.3 * r_norm + 0.3 * f_norm + 0.4 * m_norm

    # 平均订单价值
    avg_order_value = monetary_list[i] / frequency_list[i] if frequency_list[i] > 0 else 0

    # 购买强度 (订单数/生命周期天数)
    purchase_intensity = frequency_list[i] / customer_lifetime_days_list[i] if customer_lifetime_days_list[i] > 0 else 0

    # 消费趋势 (后半期消费/前半期消费)
    dates = customer_data['Date'].sort()
    if len(dates) >= 2:
        mid_date = dates[len(dates) // 2]
        first_half_monetary = customer_data.filter(pl.col('Date') <= mid_date)['Total_Amount'].sum()
        second_half_monetary = customer_data.filter(pl.col('Date') > mid_date)['Total_Amount'].sum()
        monetary_trend = second_half_monetary / first_half_monetary if first_half_monetary > 0 else 1.0
    else:
        monetary_trend = 1.0

    # Recency比率 (recency / customer_lifetime)
    recency_ratio = recency_list[i] / customer_lifetime_days_list[i] if customer_lifetime_days_list[i] > 0 else 1.0

    rfm_score_list.append(rfm_score)
    avg_order_value_list.append(avg_order_value)
    purchase_intensity_list.append(purchase_intensity)
    monetary_trend_list.append(monetary_trend)
    recency_ratio_list.append(recency_ratio)

log("✅ 创新特征计算完成")
log("")

# 构建特征DataFrame
features = pl.DataFrame({
    'Customer_ID': customer_ids,
    # RFM特征 (5个)
    'recency': recency_list,
    'frequency': frequency_list,
    'monetary': monetary_list,
    'customer_lifetime_days': customer_lifetime_days_list,
    'avg_days_between_orders': avg_days_between_orders_list,
    # 行为特征 (7个)
    'n_categories': n_categories_list,
    'avg_rating': avg_rating_list,
    'avg_delivery_days': avg_delivery_days_list,
    'avg_session': avg_session_list,
    'avg_pages': avg_pages_list,
    'discount_rate': discount_rate_list,
    'orders_last_30days': orders_last_30days_list,
    # 产品偏好特征 (4个)
    'electronics_ratio': electronics_ratio_list,
    'fashion_ratio': fashion_ratio_list,
    'home_ratio': home_ratio_list,
    'beauty_ratio': beauty_ratio_list,
    # 支付与设备特征 (3个)
    'mobile_ratio': mobile_ratio_list,
    'credit_card_ratio': credit_card_ratio_list,
    'is_returning_ratio': is_returning_ratio_list,
    # 人口统计特征 (1个)
    'age': age_list,
    # 创新特征 (5个)
    'rfm_score': rfm_score_list,
    'avg_order_value': avg_order_value_list,
    'purchase_intensity': purchase_intensity_list,
    'monetary_trend': monetary_trend_list,
    'recency_ratio': recency_ratio_list
}, strict=False)

log(f"特征工程完成:")
log(f"  - 特征数量: 25个 (不含Customer_ID)")
log(f"  - 样本数量: {len(features):,}个客户")
log("")

# ============================================================
# 阶段5: 合并特征和标签
# ============================================================
log("=" * 60)
log("阶段5: 合并特征和标签")
log("=" * 60)

# 合并特征和标签
df_final = features.join(all_customers, on='Customer_ID', how='inner')

log(f"合并后样本数: {len(df_final):,}")
log(f"特征数: 25")
log(f"标签分布:")
log(f"  - 流失 (1): {df_final.filter(pl.col('is_churn') == 1).height:,} ({df_final.filter(pl.col('is_churn') == 1).height / len(df_final) * 100:.1f}%)")
log(f"  - 未流失 (0): {df_final.filter(pl.col('is_churn') == 0).height:,} ({df_final.filter(pl.col('is_churn') == 0).height / len(df_final) * 100:.1f}%)")
log("")

# ============================================================
# 阶段6: 特征Winsorization (99%)
# ============================================================
log("=" * 60)
log("阶段6: 特征Winsorization (99%)")
log("=" * 60)

feature_cols = [col for col in df_final.columns if col not in ['Customer_ID', 'is_churn']]

for col in feature_cols:
    lower = df_final[col].quantile(0.005)
    upper = df_final[col].quantile(0.995)
    df_final = df_final.with_columns(
        pl.col(col).clip(lower, upper)
    )

log("✅ Winsorization完成")
log("")

# ============================================================
# 阶段7: 数据分割与预处理
# ============================================================
log("=" * 60)
log("阶段7: 数据分割与预处理")
log("=" * 60)

# 转换为numpy数组
X = df_final.select(feature_cols).to_numpy()
y = df_final['is_churn'].to_numpy()

# 分割训练集和测试集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log(f"训练集: {len(X_train):,}个样本")
log(f"  - 流失: {np.sum(y_train == 1):,} ({np.sum(y_train == 1) / len(y_train) * 100:.1f}%)")
log(f"  - 未流失: {np.sum(y_train == 0):,} ({np.sum(y_train == 0) / len(y_train) * 100:.1f}%)")
log(f"测试集: {len(X_test):,}个样本")
log(f"  - 流失: {np.sum(y_test == 1):,} ({np.sum(y_test == 1) / len(y_test) * 100:.1f}%)")
log(f"  - 未流失: {np.sum(y_test == 0):,} ({np.sum(y_test == 0) / len(y_test) * 100:.1f}%)")
log("")

# 标准化
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log("✅ 数据分割与标准化完成")
log("")

# ============================================================
# 阶段8: SMOTE处理类别不平衡
# ============================================================
log("=" * 60)
log("阶段8: SMOTE处理类别不平衡")
log("=" * 60)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

log(f"SMOTE前训练集: {len(X_train_scaled):,}个样本")
log(f"  - 流失: {np.sum(y_train == 1):,} ({np.sum(y_train == 1) / len(y_train) * 100:.1f}%)")
log(f"  - 未流失: {np.sum(y_train == 0):,} ({np.sum(y_train == 0) / len(y_train) * 100:.1f}%)")
log(f"SMOTE后训练集: {len(X_train_resampled):,}个样本")
log(f"  - 流失: {np.sum(y_train_resampled == 1):,} ({np.sum(y_train_resampled == 1) / len(y_train_resampled) * 100:.1f}%)")
log(f"  - 未流失: {np.sum(y_train_resampled == 0):,} ({np.sum(y_train_resampled == 0) / len(y_train_resampled) * 100:.1f}%)")
log("✅ SMOTE完成")
log("")

# ============================================================
# 阶段9: 模型训练
# ============================================================
log("=" * 60)
log("阶段9: 模型训练")
log("=" * 60)
log("")

# 9.1 Logistic Regression
log("=" * 60)
log("9.1 Logistic Regression")
log("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_resampled, y_train_resampled)

y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

log(f"测试集 - AUC: {auc_lr:.4f}, Accuracy: {acc_lr:.4f}, Precision: {precision_lr:.4f}, Recall: {recall_lr:.4f}, F1: {f1_lr:.4f}")
log("")

# 9.2 Random Forest
log("=" * 60)
log("9.2 Random Forest")
log("=" * 60)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)

y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

log(f"测试集 - AUC: {auc_rf:.4f}, Accuracy: {acc_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, F1: {f1_rf:.4f}")
log("")

# 9.3 XGBoost (带GridSearchCV)
log("=" * 60)
log("9.3 XGBoost (带GridSearchCV)")
log("=" * 60)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
grid_search_xgb = GridSearchCV(
    xgb, param_grid_xgb, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
)

log("开始超参数搜索 (3-Fold CV)...")
grid_search_xgb.fit(X_train_resampled, y_train_resampled)

log(f"最佳参数: {grid_search_xgb.best_params_}")
log(f"最佳CV得分 (AUC): {grid_search_xgb.best_score_:.4f}")
log("")

xgb_best = grid_search_xgb.best_estimator_
y_pred_xgb = xgb_best.predict(X_test_scaled)
y_pred_proba_xgb = xgb_best.predict_proba(X_test_scaled)[:, 1]

auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

log(f"测试集 - AUC: {auc_xgb:.4f}, Accuracy: {acc_xgb:.4f}, Precision: {precision_xgb:.4f}, Recall: {recall_xgb:.4f}, F1: {f1_xgb:.4f}")
log("")

# 9.4 LightGBM
log("=" * 60)
log("9.4 LightGBM")
log("=" * 60)

lgbm = LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, verbose=-1)
lgbm.fit(X_train_resampled, y_train_resampled)

y_pred_lgbm = lgbm.predict(X_test_scaled)
y_pred_proba_lgbm = lgbm.predict_proba(X_test_scaled)[:, 1]

auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
precision_lgbm = precision_score(y_test, y_pred_lgbm)
recall_lgbm = recall_score(y_test, y_pred_lgbm)
f1_lgbm = f1_score(y_test, y_pred_lgbm)

log(f"测试集 - AUC: {auc_lgbm:.4f}, Accuracy: {acc_lgbm:.4f}, Precision: {precision_lgbm:.4f}, Recall: {recall_lgbm:.4f}, F1: {f1_lgbm:.4f}")
log("")

# 9.5 Voting Classifier (集成)
log("=" * 60)
log("9.5 Voting Classifier (集成)")
log("=" * 60)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb_best),
        ('lgbm', lgbm)
    ],
    voting='soft'
)

log("训练Voting Classifier...")
voting_clf.fit(X_train_resampled, y_train_resampled)

y_pred_voting = voting_clf.predict(X_test_scaled)
y_pred_proba_voting = voting_clf.predict_proba(X_test_scaled)[:, 1]

auc_voting = roc_auc_score(y_test, y_pred_proba_voting)
acc_voting = accuracy_score(y_test, y_pred_voting)
precision_voting = precision_score(y_test, y_pred_voting)
recall_voting = recall_score(y_test, y_pred_voting)
f1_voting = f1_score(y_test, y_pred_voting)

log(f"测试集 - AUC: {auc_voting:.4f}, Accuracy: {acc_voting:.4f}, Precision: {precision_voting:.4f}, Recall: {recall_voting:.4f}, F1: {f1_voting:.4f}")
log("")

# ============================================================
# 阶段10: 模型性能对比
# ============================================================
log("=" * 60)
log("阶段10: 模型性能对比")
log("=" * 60)

results = pl.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Voting Classifier'],
    'AUC': [auc_lr, auc_rf, auc_xgb, auc_lgbm, auc_voting],
    'Accuracy': [acc_lr, acc_rf, acc_xgb, acc_lgbm, acc_voting],
    'Precision': [precision_lr, precision_rf, precision_xgb, precision_lgbm, precision_voting],
    'Recall': [recall_lr, recall_rf, recall_xgb, recall_lgbm, recall_voting],
    'F1-Score': [f1_lr, f1_rf, f1_xgb, f1_lgbm, f1_voting]
})

log(results)
log("")

# 找出最佳模型
best_model_idx = results['AUC'].arg_max()
best_model_name = results['Model'][best_model_idx]
best_auc = results['AUC'][best_model_idx]

log(f"🏆 最佳模型: {best_model_name} (AUC={best_auc:.4f})")
log("")

# ============================================================
# 阶段11: 5-Fold交叉验证 (最佳模型)
# ============================================================
log("=" * 60)
log("阶段11: 5-Fold交叉验证 (最佳模型)")
log("=" * 60)

# 使用XGBoost进行交叉验证
cv_scores = cross_val_score(
    xgb_best,
    X_train_resampled,
    y_train_resampled,
    cv=5,
    scoring='roc_auc'
)

log(f"5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
log(f"各Fold AUC: {cv_scores}")
log("")

# ============================================================
# 阶段12: 混淆矩阵 (最佳模型)
# ============================================================
log("=" * 60)
log("阶段12: 混淆矩阵 (最佳模型)")
log("=" * 60)

cm = confusion_matrix(y_test, y_pred_xgb)
log("混淆矩阵:")
log(f"              预测未流失  预测流失")
log(f"实际未流失    {cm[0][0]:>6}      {cm[0][1]:>6}")
log(f"实际流失      {cm[1][0]:>6}      {cm[1][1]:>6}")
log("")

log("分类报告:")
log(classification_report(y_test, y_pred_xgb, target_names=['未流失', '流失']))
log("")

# ============================================================
# 阶段13: 特征重要性分析 (XGBoost)
# ============================================================
log("=" * 60)
log("阶段13: 特征重要性分析 (XGBoost)")
log("=" * 60)

feature_importance = xgb_best.feature_importances_
feature_names = feature_cols

# 创建特征重要性DataFrame
importance_df = pl.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort('Importance', descending=True)

log("Top 10重要特征:")
log(importance_df.head(10))
log("")

# ============================================================
# 项目总结
# ============================================================
log("=" * 60)
log("项目总结")
log("=" * 60)
log("")

log("✅ 核心成果:")
log("  1. 特征数量: 25个 (RFM 5 + 行为 7 + 产品偏好 4 + 支付设备 3 + 人口统计 1 + 创新 5)")
log("  2. 数据处理: 99.5%截尾 + 99% Winsorization + SMOTE")
log("  3. 模型: LR + RF + XGBoost + LightGBM + Voting")
log("  4. 超参数优化: GridSearchCV (3-Fold CV)")
log("  5. 集成方法: Voting Classifier (Soft Voting)")
log("")

log("✅ 最终效果:")
log(f"  - 最佳模型: {best_model_name}")
log(f"  - 测试集AUC: {best_auc:.4f}")
log(f"  - 5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
log("")

if best_auc >= 0.75:
    log("🎉 达到预期目标 (AUC >= 0.75)!")
elif best_auc >= 0.70:
    log("✅ 接近预期目标 (AUC >= 0.70)")
else:
    log("⚠️ 未达到预期目标 (AUC < 0.70)")
log("")

log("业务洞察:")
log(f"  - Top 3特征: {importance_df['Feature'][0]}, {importance_df['Feature'][1]}, {importance_df['Feature'][2]}")
log(f"  - 流失率: {df_final.filter(pl.col('is_churn') == 1).height / len(df_final) * 100:.1f}%")
log(f"  - 模型可以识别 {recall_xgb * 100:.1f}% 的流失客户")
log(f"  - 预测为流失的客户中，{precision_xgb * 100:.1f}% 确实会流失")
log("")

log("下一步建议:")
log("  1. 基于流失预测结果，制定客户挽回策略")
log("  2. 针对高风险客户，提供个性化优惠")
log("  3. 分析流失客户的共同特征，优化产品和服务")
log("  4. 定期重新训练模型，保持预测准确性")
log("")

log("=" * 60)
log("程序执行完成")
log("=" * 60)
log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")
log("日志已保存至: eco_churn_prediction_log.txt")

