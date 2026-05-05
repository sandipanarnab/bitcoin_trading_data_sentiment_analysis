import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("BITCOIN SENTIMENT & TRADER ANALYSIS")
print("=" * 60)

# 1. DATA LOADING
print("\n[1] LOADING DATA...")
fear_greed_df = pd.read_csv('/workspace/user_input_files/fear_greed_index.csv')
trader_df = pd.read_csv('/workspace/user_input_files/historical_data_sample.csv')
print(f"   Fear/Greed: {len(fear_greed_df)} records")
print(f"   Trader: {len(trader_df)} records")

# 2. DATA CLEANING
print("\n[2] CLEANING DATA...")
fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'], format='%m/%d/%Y')
trader_df['trade_date'] = pd.to_datetime(trader_df['Timestamp IST'], format='%m/%d/%Y %H:%M', errors='coerce')
trader_df['Closed PnL'] = pd.to_numeric(trader_df['Closed PnL'], errors='coerce')
trader_df['Side'] = trader_df['Side'].str.upper().str.strip()
trader_df = trader_df.dropna(subset=['Closed PnL', 'trade_date'])
print(f"   After cleaning: {len(trader_df)} trader records")

# Create profit flag
trader_df['profit_flag'] = (trader_df['Closed PnL'] > 0).astype(int)

# Binary sentiment
sentiment_binary = {'Extreme Fear': 0, 'Fear': 0, 'Neutral': 1, 'Greed': 1, 'Extreme Greed': 1}
fear_greed_df['sentiment_binary'] = fear_greed_df['classification'].map(sentiment_binary)

# 3. DATA ALIGNMENT
print("\n[3] ALIGNING DATA...")
trader_df['trade_date_only'] = trader_df['trade_date'].dt.date
trader_df['trade_date_only'] = pd.to_datetime(trader_df['trade_date_only'])
fg_for_merge = fear_greed_df[['date', 'value', 'classification', 'sentiment_binary']].copy()
fg_for_merge.rename(columns={'date': 'trade_date_only'}, inplace=True)
merged_df = trader_df.merge(fg_for_merge, on='trade_date_only', how='left')
merged_df = merged_df.dropna(subset=['classification'])
print(f"   Merged: {len(merged_df)} trades with sentiment")

# 4. FEATURE ENGINEERING
print("\n[4] ENGINEERING FEATURES...")
merged_df['trade_size'] = merged_df['Size USD'].abs()
merged_df['is_buy'] = (merged_df['Side'] == 'BUY').astype(int)
merged_df['leverage_ratio'] = np.where(
    merged_df['Start Position'] > 0,
    merged_df['Size Tokens'] / merged_df['Start Position'],
    np.nan
)
merged_df['leverage_ratio'] = merged_df['leverage_ratio'].clip(upper=50, lower=0.01)

sentiment_map = {'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4}
merged_df['sentiment_encoded'] = merged_df['classification'].map(sentiment_map)
merged_df['is_fear'] = (merged_df['sentiment_binary'] == 0).astype(int)
merged_df['day_of_week'] = merged_df['trade_date'].dt.dayofweek
merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)

# Trader-level features
trader_stats = merged_df.groupby('Account').agg({
    'profit_flag': 'mean',
    'leverage_ratio': 'mean',
    'trade_size': 'mean',
    'Closed PnL': 'sum',
    'Size USD': 'count'
}).reset_index()
trader_stats.columns = ['Account', 'trader_win_rate', 'trader_avg_leverage', 'trader_avg_size', 'trader_total_pnl', 'trader_num_trades']
merged_df = merged_df.merge(trader_stats, on='Account', how='left')

print(f"   Features created: {len(merged_df.columns)} columns")

# 5. EDA & STATISTICS
print("\n[5] EDA & STATISTICS...")
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

# Win rates by sentiment
win_rates = merged_df.groupby('classification')['profit_flag'].mean()
win_rates = win_rates.reindex(sentiment_order)

# Fear vs Greed stats
fear_wr = merged_df[merged_df['is_fear'] == 1]['profit_flag'].mean() * 100
greed_wr = merged_df[merged_df['is_fear'] == 0]['profit_flag'].mean() * 100
print(f"   Fear win rate: {fear_wr:.1f}%")
print(f"   Greed win rate: {greed_wr:.1f}%")

# Leverage impact
high_lev_wr = merged_df[merged_df['leverage_ratio'] >= 5]['profit_flag'].mean() * 100
low_lev_wr = merged_df[merged_df['leverage_ratio'] < 5]['profit_flag'].mean() * 100
print(f"   High leverage win rate: {high_lev_wr:.1f}%")
print(f"   Low leverage win rate: {low_lev_wr:.1f}%")

# Statistical tests
fear_pnl = merged_df[merged_df['is_fear'] == 1]['Closed PnL']
greed_pnl = merged_df[merged_df['is_fear'] == 0]['Closed PnL']
stat, p_value = mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')
print(f"\n   Mann-Whitney U test p-value: {p_value:.6f}")
print(f"   Statistical significance: {'YES' if p_value < 0.05 else 'NO'}")

# Effect size
def cohens_d(g1, g2):
    n1, n2, var1, var2 = len(g1), len(g2), g1.var(), g2.var()
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled
d = cohens_d(fear_pnl, greed_pnl)
print(f"   Cohen's d effect size: {d:.4f}")

# 6. MODELING
print("\n[6] BUILDING MODELS...")
feature_cols = ['leverage_ratio', 'trade_size', 'is_buy', 'trader_win_rate',
                'trader_avg_leverage', 'trader_num_trades', 'sentiment_encoded',
                'is_fear', 'day_of_week', 'is_weekend']
target_col = 'profit_flag'

model_df = merged_df[feature_cols + [target_col]].dropna()
X = model_df[feature_cols]
y = model_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Predictions
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Metrics
lr_auc = roc_auc_score(y_test, y_prob_lr)
rf_auc = roc_auc_score(y_test, y_prob_rf)
print(f"   Logistic Regression ROC-AUC: {lr_auc:.4f}")
print(f"   Random Forest ROC-AUC: {rf_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sentiment_importance = feature_importance[feature_importance['Feature'].str.contains('sentiment|fear')]['Importance'].sum()
behavior_importance = feature_importance[~feature_importance['Feature'].str.contains('sentiment|fear')]['Importance'].sum()
print(f"\n   Sentiment feature importance: {sentiment_importance*100:.1f}%")
print(f"   Behavior feature importance: {behavior_importance*100:.1f}%")

# 7. CREATE VISUALIZATIONS
print("\n[7] CREATING VISUALIZATIONS...")
plt.style.use('seaborn-v0_8-whitegrid')

# Chart 1: Win rate by sentiment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# PnL by sentiment
sentiment_stats = merged_df.groupby('classification')['Closed PnL'].agg(['mean', 'median', 'count'])
sentiment_stats = sentiment_stats.reindex(sentiment_order)
x = range(len(sentiment_order))
axes[0,0].bar(x, win_rates.values * 100, color=['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850'], edgecolor='black')
axes[0,0].axhline(y=50, color='gray', linestyle='--')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(sentiment_order, rotation=45, ha='right')
axes[0,0].set_ylabel('Win Rate (%)')
axes[0,0].set_title('Win Rate by Market Sentiment')

# Feature importance
feature_imp_sorted = feature_importance.sort_values('Importance', ascending=True)
colors = ['#d73027' if 'sentiment' in f or 'fear' in f else 'steelblue' for f in feature_imp_sorted['Feature']]
axes[0,1].barh(feature_imp_sorted['Feature'], feature_imp_sorted['Importance'], color=colors)
axes[0,1].set_xlabel('Importance')
axes[0,1].set_title('Random Forest Feature Importance\n(Red = Sentiment)')

# PnL distribution
axes[1,0].hist(merged_df['Closed PnL'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1,0].axvline(x=0, color='red', linestyle='--', label='Break-even')
axes[1,0].set_xlabel('Closed PnL ($)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('PnL Distribution')
axes[1,0].legend()

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
axes[1,1].plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC={lr_auc:.3f})', linewidth=2)
axes[1,1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_auc:.3f})', linewidth=2)
axes[1,1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1,1].set_xlabel('False Positive Rate')
axes[1,1].set_ylabel('True Positive Rate')
axes[1,1].set_title('ROC Curves')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('/workspace/charts/analysis_summary.png', dpi=150, bbox_inches='tight')
print("   Saved: /workspace/charts/analysis_summary.png")

# 8. GENERATE KEY INSIGHTS
print("\n[8] KEY INSIGHTS:")
print("-" * 50)
print(f"1. WIN RATE: Fear={fear_wr:.1f}%, Greed={greed_wr:.1f}% (diff: {abs(fear_wr-greed_wr):.1f}%)")
print(f"2. LEVERAGE: High leverage reduces win rate by {low_lev_wr-high_lev_wr:.1f}%")
print(f"3. STATISTICAL SIGNIFICANCE: p={p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not significant'})")
print(f"4. EFFECT SIZE: Cohen's d={d:.4f} ({'Small' if abs(d)<0.5 else 'Medium' if abs(d)<0.8 else 'Large'})")
print(f"5. SENTIMENT IMPORTANCE: {sentiment_importance*100:.1f}% of total importance")
print(f"6. BEHAVIOR IMPORTANCE: {behavior_importance*100:.1f}% of total importance")
print(f"7. BEST MODEL: {'Random Forest' if rf_auc > lr_auc else 'Logistic Regression'} (AUC: {max(rf_auc, lr_auc):.4f})")
print(f"8. TOP FEATURES: {', '.join(feature_importance.head(3)['Feature'].tolist())}")

# Save model metrics
metrics_data = {
    'Metric': ['ROC-AUC', 'F1-Score', 'Precision', 'Recall'],
    'Logistic_Regression': [
        f'{lr_auc:.4f}',
        f'{f1_score(y_test, lr_model.predict(X_test_scaled)):.4f}',
        f'{precision_score(y_test, lr_model.predict(X_test_scaled)):.4f}',
        f'{recall_score(y_test, lr_model.predict(X_test_scaled)):.4f}'
    ],
    'Random_Forest': [
        f'{rf_auc:.4f}',
        f'{f1_score(y_test, rf_model.predict(X_test)):.4f}',
        f'{precision_score(y_test, rf_model.predict(X_test)):.4f}',
        f'{recall_score(y_test, rf_model.predict(X_test)):.4f}'
    ]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('/workspace/models/metrics.csv', index=False)
print(f"\n   Metrics saved to: /workspace/models/metrics.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)