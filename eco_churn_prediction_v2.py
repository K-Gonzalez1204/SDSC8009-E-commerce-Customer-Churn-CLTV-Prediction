"""
电商客户流失预测项目 - 优化版本 V2
基于V1的运行结果进行5项优化
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
import sys
import logging

# 机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# 配置日志系统 (双输出: 控制台 + 文件)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('eco_churn_prediction_v2_log.txt', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def log(message):
    """统一的日志输出函数"""
    logging.info(message)

# ============================================================
# 项目信息
# ============================================================
log("=" * 60)
log("电商客户流失预测项目 - 优化版本 V2")
log("=" * 60)
log(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("")

log("优化内容:")
log("1. 调整时间窗口: 特征期9个月 vs 标签期5个月")
log("2. 调整流失定义: 标签期内无购买 且 recency > 90天")
log("3. 增加5个新特征: 周期性、趋势、集中度、敏感度、参与度")
log("4. 添加CatBoost模型")
log("5. 特征选择优化: SelectKBest (Top 20)")
log("")

# ============================================================
# 阶段1: 数据加载
# ============================================================
log("=" * 60)
log("阶段1: 数据加载")
log("=" * 60)

df = pl.read_csv('data/ecommerce_customer_behavior_dataset_v2.csv')
df = df.with_columns(
    pl.col('Date').str.to_date('%Y-%m-%d')
)

log(f"数据加载完成:")
log(f"  - 总记录数: {df.height:,}")
log(f"  - 总客户数: {df['Customer_ID'].n_unique():,}")
log(f"  - 时间范围: {df['Date'].min()} 至 {df['Date'].max()}")
log("")

# ============================================================
# 阶段2: 数据清洗 (异常值处理)
# ============================================================
log("=" * 60)
log("阶段2: 数据清洗 (异常值处理)")
log("=" * 60)

# 99.5%分位数截尾
total_amount_99_5 = df['Total_Amount'].quantile(0.995)
unit_price_99_5 = df['Unit_Price'].quantile(0.995)

log(f"Total_Amount 99.5%分位数: {total_amount_99_5:.2f}")
log(f"Unit_Price 99.5%分位数: {unit_price_99_5:.2f}")
log("")

before_count = df.height
df = df.filter(
    (pl.col('Total_Amount') <= total_amount_99_5) &
    (pl.col('Unit_Price') <= unit_price_99_5)
)
after_count = df.height

log(f"删除前: {before_count:,}条记录")
log(f"删除后: {after_count:,}条记录")
log(f"删除比例: {(before_count - after_count) / before_count * 100:.2f}%")
log("✅ 数据清洗完成")
log("")

# ============================================================
# 阶段3: 流失标签构建 (优化版)
# ============================================================
log("=" * 60)
log("阶段3: 流失标签构建 (优化版)")
log("=" * 60)

# 优化1: 调整时间窗口
cutoff_date = datetime(2023, 10, 1).date()  # 从2023-09-01改为2023-10-01
feature_start = datetime(2023, 1, 1).date()
label_end = datetime(2024, 3, 1).date()

log(f"Cutoff Date: {cutoff_date}")
log(f"特征期: {feature_start} 至 {cutoff_date} (9个月)")  # 从8个月改为9个月
log(f"标签期: {cutoff_date} 至 {label_end} (5个月)")  # 从6个月改为5个月
log("")

# 特征期数据
df_feature = df.filter(pl.col('Date') < cutoff_date)

# 标签期数据
df_label = df.filter(
    (pl.col('Date') >= cutoff_date) & 
    (pl.col('Date') < label_end)
)

# 特征期有购买的客户
feature_customers = df_feature['Customer_ID'].unique().to_list()

# 标签期有购买的客户
label_customers = df_label['Customer_ID'].unique().to_list()

# 优化2: 调整流失定义
# 计算每个客户在特征期的最后购买日期
last_purchase = df_feature.group_by('Customer_ID').agg(
    pl.col('Date').max().alias('last_purchase_date')
)

# 流失定义: 标签期内无购买 且 距离cutoff_date超过90天
churn_threshold_days = 90
churn_customers = []

for customer_id in feature_customers:
    # 检查是否在标签期购买
    if customer_id not in label_customers:
        # 检查recency
        last_date = last_purchase.filter(pl.col('Customer_ID') == customer_id)['last_purchase_date'][0]
        recency_days = (cutoff_date - last_date).days
        if recency_days > churn_threshold_days:
            churn_customers.append(customer_id)

# 构建标签DataFrame
all_customers = pl.DataFrame({
    'Customer_ID': feature_customers,
    'is_churn': [1 if cid in churn_customers else 0 for cid in feature_customers]
})

log(f"标签统计:")
log(f"  - 总客户数: {len(feature_customers):,}")
log(f"  - 流失客户数: {len(churn_customers):,} ({len(churn_customers)/len(feature_customers)*100:.1f}%)")
log(f"  - 未流失客户数: {len(feature_customers) - len(churn_customers):,} ({(len(feature_customers) - len(churn_customers))/len(feature_customers)*100:.1f}%)")
log("")

# ============================================================
# 阶段4: 特征工程 (30个特征 = 原25个 + 新5个)
# ============================================================
log("=" * 60)
log("阶段4: 特征工程 (30个特征)")
log("=" * 60)

# 初始化特征列表
customer_ids = []
# RFM特征 (5个)
recency_list = []
frequency_list = []
monetary_list = []
customer_lifetime_days_list = []
avg_days_between_orders_list = []

# 行为特征 (7个)
n_categories_list = []
avg_rating_list = []
avg_delivery_days_list = []
avg_session_list = []
avg_pages_list = []
discount_rate_list = []
orders_last_30days_list = []

# 产品偏好特征 (4个)
electronics_ratio_list = []
fashion_ratio_list = []
home_ratio_list = []
beauty_ratio_list = []

# 支付与设备特征 (3个)
mobile_ratio_list = []
credit_card_ratio_list = []
is_returning_ratio_list = []

# 人口统计特征 (1个)
age_list = []

# 创新特征 (5个)
rfm_score_list = []
avg_order_value_list = []
purchase_intensity_list = []
monetary_trend_list = []
recency_ratio_list = []

# 新增特征 (5个) - 优化3
purchase_regularity_list = []  # 购买周期性
recent_3m_trend_list = []  # 最近3个月消费趋势
category_concentration_list = []  # 品类集中度
price_sensitivity_list = []  # 价格敏感度
engagement_score_list = []  # 参与度得分

log("计算30个特征...")

for customer_id in feature_customers:
    customer_data = df_feature.filter(pl.col('Customer_ID') == customer_id)

    # 基本信息
    customer_ids.append(customer_id)
    n_orders = customer_data.height

    # RFM特征
    last_purchase_date = customer_data['Date'].max()
    first_purchase_date = customer_data['Date'].min()
    recency = (cutoff_date - last_purchase_date).days
    frequency = n_orders
    monetary = customer_data['Total_Amount'].sum()
    customer_lifetime = (last_purchase_date - first_purchase_date).days + 1
    avg_days_between = customer_lifetime / max(frequency - 1, 1)

    recency_list.append(recency)
    frequency_list.append(frequency)
    monetary_list.append(monetary)
    customer_lifetime_days_list.append(customer_lifetime)
    avg_days_between_orders_list.append(avg_days_between)

    # 行为特征
    n_categories = customer_data['Product_Category'].n_unique()
    avg_rating = customer_data['Customer_Rating'].mean()
    avg_delivery = customer_data['Delivery_Time_Days'].mean()
    avg_session = customer_data['Session_Duration_Minutes'].mean()
    avg_pages = customer_data['Pages_Viewed'].mean()
    total_discount = customer_data['Discount_Amount'].sum()
    discount_rate = total_discount / monetary if monetary > 0 else 0

    # 最近30天订单数
    days_30_ago = cutoff_date - timedelta(days=30)
    orders_last_30 = customer_data.filter(pl.col('Date') >= days_30_ago).height

    n_categories_list.append(n_categories)
    avg_rating_list.append(avg_rating)
    avg_delivery_days_list.append(avg_delivery)
    avg_session_list.append(avg_session)
    avg_pages_list.append(avg_pages)
    discount_rate_list.append(discount_rate)
    orders_last_30days_list.append(orders_last_30)

    # 产品偏好特征
    category_counts = customer_data.group_by('Product_Category').agg(pl.count().alias('count'))
    total_orders = n_orders

    electronics_ratio = category_counts.filter(pl.col('Product_Category') == 'Electronics')['count'][0] / total_orders if len(category_counts.filter(pl.col('Product_Category') == 'Electronics')) > 0 else 0
    fashion_ratio = category_counts.filter(pl.col('Product_Category') == 'Fashion')['count'][0] / total_orders if len(category_counts.filter(pl.col('Product_Category') == 'Fashion')) > 0 else 0
    home_ratio = category_counts.filter(pl.col('Product_Category') == 'Home & Garden')['count'][0] / total_orders if len(category_counts.filter(pl.col('Product_Category') == 'Home & Garden')) > 0 else 0
    beauty_ratio = category_counts.filter(pl.col('Product_Category') == 'Beauty & Personal Care')['count'][0] / total_orders if len(category_counts.filter(pl.col('Product_Category') == 'Beauty & Personal Care')) > 0 else 0

    electronics_ratio_list.append(electronics_ratio)
    fashion_ratio_list.append(fashion_ratio)
    home_ratio_list.append(home_ratio)
    beauty_ratio_list.append(beauty_ratio)

    # 支付与设备特征
    mobile_count = customer_data.filter(pl.col('Device_Type') == 'Mobile')['Device_Type'].count()
    mobile_ratio = mobile_count / total_orders

    credit_card_count = customer_data.filter(pl.col('Payment_Method') == 'Credit Card')['Payment_Method'].count()
    credit_card_ratio = credit_card_count / total_orders

    is_returning_count = customer_data.filter(pl.col('Is_Returning_Customer') == 1)['Is_Returning_Customer'].count()
    is_returning_ratio = is_returning_count / total_orders

    mobile_ratio_list.append(mobile_ratio)
    credit_card_ratio_list.append(credit_card_ratio)
    is_returning_ratio_list.append(is_returning_ratio)

    # 人口统计特征
    age = customer_data['Age'][0]
    age_list.append(age)

    # 创新特征
    # RFM Score (简化版)
    r_score = 5 if recency < 30 else (4 if recency < 60 else (3 if recency < 90 else (2 if recency < 180 else 1)))
    f_score = 5 if frequency >= 8 else (4 if frequency >= 6 else (3 if frequency >= 4 else (2 if frequency >= 2 else 1)))
    m_score = 5 if monetary >= 5000 else (4 if monetary >= 3000 else (3 if monetary >= 1500 else (2 if monetary >= 500 else 1)))
    rfm_score = r_score * 100 + f_score * 10 + m_score

    avg_order_value = monetary / frequency
    purchase_intensity = frequency / (customer_lifetime / 30)  # 每月购买次数

    # 消费趋势 (前半期 vs 后半期)
    mid_date = first_purchase_date + timedelta(days=customer_lifetime // 2)
    first_half_monetary = customer_data.filter(pl.col('Date') < mid_date)['Total_Amount'].sum()
    second_half_monetary = customer_data.filter(pl.col('Date') >= mid_date)['Total_Amount'].sum()
    monetary_trend = (second_half_monetary - first_half_monetary) / max(first_half_monetary, 1)

    recency_ratio = recency / customer_lifetime if customer_lifetime > 0 else 0

    rfm_score_list.append(rfm_score)
    avg_order_value_list.append(avg_order_value)
    purchase_intensity_list.append(purchase_intensity)
    monetary_trend_list.append(monetary_trend)
    recency_ratio_list.append(recency_ratio)

    # 新增特征 (5个)
    # 1. 购买周期性 (标准差/均值，越小越规律)
    if frequency > 1:
        dates = sorted(customer_data['Date'].to_list())
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        if len(intervals) > 0:
            purchase_regularity = np.std(intervals) / (np.mean(intervals) + 1)
        else:
            purchase_regularity = 0
    else:
        purchase_regularity = 0
    purchase_regularity_list.append(purchase_regularity)

    # 2. 最近3个月消费趋势
    days_90_ago = cutoff_date - timedelta(days=90)
    recent_3m_data = customer_data.filter(pl.col('Date') >= days_90_ago)
    recent_3m_monetary = recent_3m_data['Total_Amount'].sum()
    avg_monthly_monetary = monetary / (customer_lifetime / 30)
    recent_3m_trend = (recent_3m_monetary / 3) / max(avg_monthly_monetary, 1) - 1
    recent_3m_trend_list.append(recent_3m_trend)

    # 3. 品类集中度 (Herfindahl指数)
    category_shares = []
    for cat in category_counts['Product_Category']:
        count = category_counts.filter(pl.col('Product_Category') == cat)['count'][0]
        share = count / total_orders
        category_shares.append(share ** 2)
    category_concentration = sum(category_shares)
    category_concentration_list.append(category_concentration)

    # 4. 价格敏感度 (折扣使用率变化)
    if frequency > 1:
        first_half_data = customer_data.filter(pl.col('Date') < mid_date)
        second_half_data = customer_data.filter(pl.col('Date') >= mid_date)
        first_half_discount_rate = first_half_data['Discount_Amount'].sum() / max(first_half_data['Total_Amount'].sum(), 1)
        second_half_discount_rate = second_half_data['Discount_Amount'].sum() / max(second_half_data['Total_Amount'].sum(), 1)
        price_sensitivity = second_half_discount_rate - first_half_discount_rate
    else:
        price_sensitivity = 0
    price_sensitivity_list.append(price_sensitivity)

    # 5. 参与度得分 (session + pages + rating综合)
    engagement_score = (avg_session / 30) * 0.3 + (avg_pages / 10) * 0.3 + (avg_rating / 5) * 0.4
    engagement_score_list.append(engagement_score)

log("✅ 特征计算完成")
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
    'recency_ratio': recency_ratio_list,
    # 新增特征 (5个)
    'purchase_regularity': purchase_regularity_list,
    'recent_3m_trend': recent_3m_trend_list,
    'category_concentration': category_concentration_list,
    'price_sensitivity': price_sensitivity_list,
    'engagement_score': engagement_score_list
}, strict=False)

log(f"特征工程完成:")
log(f"  - 特征数量: 30个 (不含Customer_ID)")
log(f"  - 样本数量: {features.height:,}个客户")
log("")

# ============================================================
# 阶段5: 合并特征和标签
# ============================================================
log("=" * 60)
log("阶段5: 合并特征和标签")
log("=" * 60)

df_final = features.join(all_customers, on='Customer_ID', how='inner')

log(f"合并后样本数: {df_final.height:,}")
log(f"特征数: 30")
log(f"标签分布:")
churn_count = df_final.filter(pl.col('is_churn') == 1).height
non_churn_count = df_final.filter(pl.col('is_churn') == 0).height
log(f"  - 流失 (1): {churn_count:,} ({churn_count/df_final.height*100:.1f}%)")
log(f"  - 未流失 (0): {non_churn_count:,} ({non_churn_count/df_final.height*100:.1f}%)")
log("")

# ============================================================
# 阶段6: 特征Winsorization (99%)
# ============================================================
log("=" * 60)
log("阶段6: 特征Winsorization (99%)")
log("=" * 60)

feature_cols = [col for col in df_final.columns if col not in ['Customer_ID', 'is_churn']]

for col in feature_cols:
    lower = df_final[col].quantile(0.01)
    upper = df_final[col].quantile(0.99)
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

X = df_final.select(feature_cols).to_numpy()
y = df_final['is_churn'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log(f"训练集: {X_train.shape[0]:,}个样本")
log(f"  - 流失: {np.sum(y_train == 1):,} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
log(f"  - 未流失: {np.sum(y_train == 0):,} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
log(f"测试集: {X_test.shape[0]:,}个样本")
log(f"  - 流失: {np.sum(y_test == 1):,} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
log(f"  - 未流失: {np.sum(y_test == 0):,} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
log("")

# 标准化
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log("✅ 数据分割与标准化完成")
log("")

# ============================================================
# 阶段8: 特征选择 (优化5)
# ============================================================
log("=" * 60)
log("阶段8: 特征选择 (SelectKBest, Top 20)")
log("=" * 60)

# 使用mutual_info_classif选择Top 20特征
selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# 获取选中的特征名
selected_features_mask = selector.get_support()
selected_features = [feature_cols[i] for i, selected in enumerate(selected_features_mask) if selected]

log(f"原始特征数: 30")
log(f"选择特征数: 20")
log(f"选中的特征:")
for i, feat in enumerate(selected_features, 1):
    log(f"  {i}. {feat}")
log("")

# ============================================================
# 阶段9: SMOTE处理类别不平衡
# ============================================================
log("=" * 60)
log("阶段9: SMOTE处理类别不平衡")
log("=" * 60)

log(f"SMOTE前训练集: {X_train_selected.shape[0]:,}个样本")
log(f"  - 流失: {np.sum(y_train == 1):,} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
log(f"  - 未流失: {np.sum(y_train == 0):,} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

log(f"SMOTE后训练集: {X_train_resampled.shape[0]:,}个样本")
log(f"  - 流失: {np.sum(y_train_resampled == 1):,} ({np.sum(y_train_resampled == 1)/len(y_train_resampled)*100:.1f}%)")
log(f"  - 未流失: {np.sum(y_train_resampled == 0):,} ({np.sum(y_train_resampled == 0)/len(y_train_resampled)*100:.1f}%)")
log("✅ SMOTE完成")
log("")

# ============================================================
# 阶段10: 模型训练 (6个模型，新增CatBoost)
# ============================================================
log("=" * 60)
log("阶段10: 模型训练")
log("=" * 60)
log("")

models = {}
results = []

# 10.1 Logistic Regression
log("=" * 60)
log("10.1 Logistic Regression")
log("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr.predict(X_test_selected)
y_pred_proba_lr = lr.predict_proba(X_test_selected)[:, 1]

auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

log(f"测试集 - AUC: {auc_lr:.4f}, Accuracy: {acc_lr:.4f}, Precision: {prec_lr:.4f}, Recall: {rec_lr:.4f}, F1: {f1_lr:.4f}")
log("")

models['Logistic Regression'] = lr
results.append({
    'Model': 'Logistic Regression',
    'AUC': auc_lr,
    'Accuracy': acc_lr,
    'Precision': prec_lr,
    'Recall': rec_lr,
    'F1-Score': f1_lr
})

# 10.2 Random Forest
log("=" * 60)
log("10.2 Random Forest")
log("=" * 60)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf.predict(X_test_selected)
y_pred_proba_rf = rf.predict_proba(X_test_selected)[:, 1]

auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

log(f"测试集 - AUC: {auc_rf:.4f}, Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f}, F1: {f1_rf:.4f}")
log("")

models['Random Forest'] = rf
results.append({
    'Model': 'Random Forest',
    'AUC': auc_rf,
    'Accuracy': acc_rf,
    'Precision': prec_rf,
    'Recall': rec_rf,
    'F1-Score': f1_rf
})

# 10.3 XGBoost (带GridSearchCV)
log("=" * 60)
log("10.3 XGBoost (带GridSearchCV)")
log("=" * 60)

log("开始超参数搜索 (3-Fold CV)...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(random_state=42, eval_metric='logloss')
grid_search = GridSearchCV(xgb, xgb_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
grid_search.fit(X_train_resampled, y_train_resampled)

log(f"最佳参数: {grid_search.best_params_}")
log(f"最佳CV得分 (AUC): {grid_search.best_score_:.4f}")
log("")

best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test_selected)
y_pred_proba_xgb = best_xgb.predict_proba(X_test_selected)[:, 1]

auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

log(f"测试集 - AUC: {auc_xgb:.4f}, Accuracy: {acc_xgb:.4f}, Precision: {prec_xgb:.4f}, Recall: {rec_xgb:.4f}, F1: {f1_xgb:.4f}")
log("")

models['XGBoost'] = best_xgb
results.append({
    'Model': 'XGBoost',
    'AUC': auc_xgb,
    'Accuracy': acc_xgb,
    'Precision': prec_xgb,
    'Recall': rec_xgb,
    'F1-Score': f1_xgb
})

# 10.4 LightGBM
log("=" * 60)
log("10.4 LightGBM")
log("=" * 60)

lgbm = LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, verbose=-1)
lgbm.fit(X_train_resampled, y_train_resampled)
y_pred_lgbm = lgbm.predict(X_test_selected)
y_pred_proba_lgbm = lgbm.predict_proba(X_test_selected)[:, 1]

auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
prec_lgbm = precision_score(y_test, y_pred_lgbm)
rec_lgbm = recall_score(y_test, y_pred_lgbm)
f1_lgbm = f1_score(y_test, y_pred_lgbm)

log(f"测试集 - AUC: {auc_lgbm:.4f}, Accuracy: {acc_lgbm:.4f}, Precision: {prec_lgbm:.4f}, Recall: {rec_lgbm:.4f}, F1: {f1_lgbm:.4f}")
log("")

models['LightGBM'] = lgbm
results.append({
    'Model': 'LightGBM',
    'AUC': auc_lgbm,
    'Accuracy': acc_lgbm,
    'Precision': prec_lgbm,
    'Recall': rec_lgbm,
    'F1-Score': f1_lgbm
})

# 10.5 CatBoost (新增 - 优化4)
log("=" * 60)
log("10.5 CatBoost (新增)")
log("=" * 60)

catboost = CatBoostClassifier(
    iterations=100,
    depth=8,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
catboost.fit(X_train_resampled, y_train_resampled)
y_pred_catboost = catboost.predict(X_test_selected)
y_pred_proba_catboost = catboost.predict_proba(X_test_selected)[:, 1]

auc_catboost = roc_auc_score(y_test, y_pred_proba_catboost)
acc_catboost = accuracy_score(y_test, y_pred_catboost)
prec_catboost = precision_score(y_test, y_pred_catboost)
rec_catboost = recall_score(y_test, y_pred_catboost)
f1_catboost = f1_score(y_test, y_pred_catboost)

log(f"测试集 - AUC: {auc_catboost:.4f}, Accuracy: {acc_catboost:.4f}, Precision: {prec_catboost:.4f}, Recall: {rec_catboost:.4f}, F1: {f1_catboost:.4f}")
log("")

models['CatBoost'] = catboost
results.append({
    'Model': 'CatBoost',
    'AUC': auc_catboost,
    'Accuracy': acc_catboost,
    'Precision': prec_catboost,
    'Recall': rec_catboost,
    'F1-Score': f1_catboost
})

# 10.6 Voting Classifier (集成)
log("=" * 60)
log("10.6 Voting Classifier (集成)")
log("=" * 60)

log("训练Voting Classifier...")
voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', best_xgb),
        ('lgbm', lgbm),
        ('catboost', catboost)
    ],
    voting='soft'
)
voting.fit(X_train_resampled, y_train_resampled)
y_pred_voting = voting.predict(X_test_selected)
y_pred_proba_voting = voting.predict_proba(X_test_selected)[:, 1]

auc_voting = roc_auc_score(y_test, y_pred_proba_voting)
acc_voting = accuracy_score(y_test, y_pred_voting)
prec_voting = precision_score(y_test, y_pred_voting)
rec_voting = recall_score(y_test, y_pred_voting)
f1_voting = f1_score(y_test, y_pred_voting)

log(f"测试集 - AUC: {auc_voting:.4f}, Accuracy: {acc_voting:.4f}, Precision: {prec_voting:.4f}, Recall: {rec_voting:.4f}, F1: {f1_voting:.4f}")
log("")

models['Voting Classifier'] = voting
results.append({
    'Model': 'Voting Classifier',
    'AUC': auc_voting,
    'Accuracy': acc_voting,
    'Precision': prec_voting,
    'Recall': rec_voting,
    'F1-Score': f1_voting
})

# ============================================================
# 阶段11: 模型性能对比
# ============================================================
log("=" * 60)
log("阶段11: 模型性能对比")
log("=" * 60)

results_df = pl.DataFrame(results)
print(results_df)
log("")

# 找出最佳模型
best_model_name = max(results, key=lambda x: x['AUC'])['Model']
best_auc = max(results, key=lambda x: x['AUC'])['AUC']
log(f"🏆 最佳模型: {best_model_name} (AUC={best_auc:.4f})")
log("")

# ============================================================
# 阶段12: 5-Fold交叉验证 (最佳模型)
# ============================================================
log("=" * 60)
log("阶段12: 5-Fold交叉验证 (最佳模型)")
log("=" * 60)

best_model = models[best_model_name]
cv_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc', n_jobs=-1)

log(f"5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
log(f"各Fold AUC: {cv_scores}")
log("")

# ============================================================
# 阶段13: 混淆矩阵 (最佳模型)
# ============================================================
log("=" * 60)
log("阶段13: 混淆矩阵 (最佳模型)")
log("=" * 60)

y_pred_best = best_model.predict(X_test_selected)
cm = confusion_matrix(y_test, y_pred_best)

log("混淆矩阵:")
log(f"              预测未流失  预测流失")
log(f"实际未流失       {cm[0][0]}          {cm[0][1]}")
log(f"实际流失         {cm[1][0]}          {cm[1][1]}")
log("")

log("分类报告:")
log(classification_report(y_test, y_pred_best, target_names=['未流失', '流失']))
log("")

# ============================================================
# 阶段14: 特征重要性分析
# ============================================================
log("=" * 60)
log("阶段14: 特征重要性分析 (XGBoost)")
log("=" * 60)

feature_importance = best_xgb.feature_importances_
importance_df = pl.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importance
}).sort('Importance', descending=True)

log("Top 10重要特征:")
print(importance_df.head(10))
log("")

# ============================================================
# 项目总结
# ============================================================
log("=" * 60)
log("项目总结")
log("=" * 60)
log("")

log("✅ 核心成果:")
log("  1. 优化时间窗口: 特征期9个月 vs 标签期5个月")
log("  2. 优化流失定义: 标签期内无购买 且 recency > 90天")
log("  3. 特征数量: 30个 → 20个 (SelectKBest)")
log("  4. 数据处理: 99.5%截尾 + 99% Winsorization + SMOTE")
log("  5. 模型: LR + RF + XGBoost + LightGBM + CatBoost + Voting (6个)")
log("  6. 超参数优化: GridSearchCV (3-Fold CV)")
log("  7. 集成方法: Voting Classifier (Soft Voting)")
log("")

log("✅ 最终效果:")
log(f"  - 最佳模型: {best_model_name}")
log(f"  - 测试集AUC: {best_auc:.4f}")
log(f"  - 5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
log("")

if best_auc >= 0.75:
    log("🎉 达到预期目标 (AUC >= 0.75)!")
elif best_auc >= 0.70:
    log("⚠️ 接近预期目标 (AUC >= 0.70)")
else:
    log("⚠️ 未达到预期目标 (AUC < 0.70)")
log("")

log("业务洞察:")
log(f"  - Top 3特征: {', '.join(importance_df['Feature'].head(3).to_list())}")
log(f"  - 流失率: {churn_count/df_final.height*100:.1f}%")
log(f"  - 模型可以识别 {rec_voting*100:.1f}% 的流失客户")
log(f"  - 预测为流失的客户中，{prec_voting*100:.1f}% 确实会流失")
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
log("日志已保存至: eco_churn_prediction_v2_log.txt")

