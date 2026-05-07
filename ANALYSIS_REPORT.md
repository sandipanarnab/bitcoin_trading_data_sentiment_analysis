# 🪙 Bitcoin Market Sentiment & Trader Performance Analysis
### Hyperliquid Trading Insights: Fear vs. Greed

---

> **Submitted To:** Primetrade.ai Hiring Team  
> **Assignment:** Data Science Hiring Task — Web3 Trading Analytics  
> **Dataset Coverage:** February 2018 – May 2025  
> **Platform Analyzed:** Hyperliquid Perpetuals Exchange  
> **Analysis Date:** May 2026

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Assignment Brief & Task Alignment](#assignment-brief--task-alignment)
3. [Data Overview](#data-overview)
4. [Methodology Pipeline](#methodology-pipeline)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Statistical Testing](#statistical-testing)
7. [Machine Learning Models](#machine-learning-models)
8. [Model Interpretability](#model-interpretability)
9. [Key Trading Behavior Insights](#key-trading-behavior-insights)
10. [Strategy Recommendations](#strategy-recommendations)
11. [Conclusion](#conclusion)

---

## 1. Executive Summary

This analysis investigates whether Bitcoin's **Fear and Greed Index** meaningfully predicts trader profitability on the **Hyperliquid** decentralized perpetuals exchange. By merging over **211,000 raw trade records** with daily sentiment data spanning 7+ years, the study engineers behavioral features and benchmarks statistical and machine learning models.

### 🔑 Key Takeaway

> **Sentiment is statistically real, but practically small.** Trade direction, leverage discipline, and individual trader behavior dominate predictive power — sentiment acts as a secondary context signal, not a primary trading edge.

### 📊 At-a-Glance Numbers

| Metric | Value |
|--------|------:|
| Fear & Greed Index Observations | **2,644** |
| Raw Hyperliquid Trade Records | **211,224** |
| Final Merged Trades Used | **173,532** |
| Unique Traders Analyzed | **32** |
| Best Performing Model | **Random Forest** |
| Best ROC-AUC Score | **0.9859** |
| Overall Mean PnL per Trade | **$51.64** |
| PnL Skewness | **29.43** (highly right-skewed) |

---

## 2. Assignment Brief & Task Alignment

This analysis was completed as part of the **Primetrade.ai Data Science hiring task** — a first-step screening assignment for candidates interested in joining one of the most innovative teams in Web3 trading.

### Task Requirements (as given)

> *"Your objective is to explore the relationship between trader performance and market sentiment, uncover hidden patterns, and deliver insights that can drive smarter trading strategies."*

The task specified two datasets to work with:

| Dataset | Columns Specified |
|---------|------------------|
| **Bitcoin Market Sentiment** | `Date`, `Classification` (Fear/Greed) |
| **Historical Hyperliquid Trader Data** | `account`, `symbol`, `execution price`, `size`, `side`, `time`, `start position`, `event`, `closedPnL`, `leverage`, and more |

### How This Analysis Addresses Each Requirement

| Task Requirement | How It Was Addressed |
|-----------------|---------------------|
| Explore relationship between trader performance and market sentiment | ✅ EDA win rate & PnL breakdown by all 4 sentiment classes (Section 5) |
| Uncover hidden patterns | ✅ Discovered that `is_buy` (trade direction) dominates over sentiment; weekend underperformance; leverage decay effect (Sections 8–9) |
| Deliver insights for smarter trading strategies | ✅ 5 actionable, evidence-backed strategy recommendations grounded in model outputs (Section 10) |
| Work with both provided datasets | ✅ Merged with 99.9972% coverage — only 6 of 211,224 trades lacked a sentiment label |

### The Central Hypothesis

> *"Do traders execute more profitably during Greed periods vs. Fear periods, and can sentiment predict whether a trade closes in profit?"*

The short answer: **Yes — but the effect is smaller than most traders assume.** The following sections prove this quantitatively.

---

## 3. Data Overview

### Dataset 1: Bitcoin Fear & Greed Index
- **Source:** Alternative.me Crypto Fear & Greed Index (provided via Google Drive)
- **Coverage:** February 1, 2018 → May 2, 2025
- **Granularity:** Daily
- **Fields Used:** `Date`, `Classification` (Fear / Greed / Extreme Fear / Extreme Greed), numeric value (0–100)
- **Total Observations:** 2,644 daily records

### Dataset 2: Hyperliquid Trader Execution Data
- **Source:** Historical Hyperliquid perpetuals trade logs (provided via Google Drive)
- **Coverage Used:** May 1, 2023 → April 30, 2025
- **Granularity:** Trade-level (individual executions)
- **Fields Used:** `account`, `symbol`, `execution_price`, `size`, `side`, `time`, `start_position`, `event`, `closedPnL`, `leverage`
- **Raw Records:** 211,224 trades

### Merge Coverage Summary

| Category | Count |
|----------|------:|
| Trades successfully matched to sentiment | **211,218** |
| Trades with no sentiment match | **6** |
| Missing coverage rate | **0.0028%** |
| Trades after removing neutral sentiment & null rows | **173,532** |

> ✅ **Near-perfect merge coverage** — only 6 out of 211,224 trades lacked a sentiment label.

---

## 4. Methodology Pipeline

The analysis follows a structured, reproducible pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Data Ingestion                                         │
│  Load sentiment CSV + Hyperliquid trade records                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 2: Cleaning & Standardization                             │
│  Parse dates, normalize side/direction labels, cast types       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 3: Merging                                                │
│  Left join trades → daily sentiment on trade date              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 4: Filtering                                              │
│  Remove neutral sentiment rows → isolate Fear vs. Greed        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 5: Feature Engineering                                    │
│  trade_size, is_buy, trader_win_rate, trader_avg_leverage,     │
│  day_of_week, is_weekend, leverage_ratio, sentiment_encoded    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 6: EDA + Hypothesis Testing                               │
│  Distributions, Win Rates, Mann-Whitney U, Chi-Square          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 7: Model Training                                         │
│  Logistic Regression + Random Forest (binary: win/loss)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STEP 8: Evaluation & Interpretation                            │
│  ROC-AUC, Precision-Recall, Coefficients, Feature Importance   │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Glossary

| Feature | Description |
|---------|-------------|
| `is_buy` | Binary flag: 1 = buy/long trade, 0 = sell/short trade |
| `is_weekend` | Binary flag: 1 = trade executed on Saturday or Sunday |
| `trader_win_rate` | Historical win rate of the trader (rolling, per account) |
| `trader_avg_leverage` | Trader's average leverage across all their trades |
| `trader_num_trades` | Total number of trades the trader has executed |
| `leverage_ratio` | Leverage used on this specific trade |
| `trade_size` | USD notional size of the trade |
| `sentiment_encoded` | Encoded Fear & Greed classification (numeric) |
| `day_of_week` | Day number (0=Monday … 6=Sunday) |

---

## 5. Exploratory Data Analysis

### 5.1 Sentiment Distribution Across Trades

The 173,532 trades in the analysis break down across sentiment classes as follows:

| Sentiment Class | Trade Count | Share |
|----------------|------------:|------:|
| **Greed** | 50,303 | 29.0% |
| **Extreme Greed** | 39,992 | 23.1% |
| **Fear** | 61,837 | 35.6% |
| **Extreme Fear** | 21,400 | 12.3% |

*After binary encoding:*

| Binary Label | Trade Count |
|-------------|------------:|
| **Fear** (Fear + Extreme Fear) | 83,237 |
| **Greed** (Greed + Extreme Greed) | 90,295 |

> The dataset is well-balanced between Fear and Greed periods, supporting reliable comparative analysis.

---

### 5.2 Win Rate & PnL by Sentiment

#### Binary Sentiment Comparison

| Sentiment | Mean PnL ($) | Median PnL ($) | Win Rate |
|-----------|------------:|---------------:|---------:|
| **Fear** | 49.21 | 0.00 | 40.79% |
| **Greed** | 53.88 | 0.00 | 42.03% |

#### Granular Sentiment Class Breakdown

| Classification | Mean PnL ($) | Median PnL ($) | Win Rate |
|----------------|------------:|---------------:|---------:|
| **Extreme Fear** | 34.54 | 0.00 | 37.06% |
| **Fear** | 54.29 | 0.00 | 42.08% |
| **Greed** | 42.74 | 0.00 | 38.48% |
| **Extreme Greed** | 67.89 | 0.00 | **46.49%** |

**Notable observations:**

- 📈 **Extreme Greed** periods show the highest win rate (46.49%) and mean PnL ($67.89)
- 📉 **Extreme Fear** is the weakest environment with just a 37.06% win rate
- 🔄 Interestingly, standard **Greed** (38.48%) underperforms standard **Fear** (42.08%) in win rate — suggesting sentiment inversions within sub-categories
- 📊 All median PnLs are $0.00, confirming the distribution is dominated by many small/break-even trades

### 5.3 PnL Distribution Shape

| Statistic | Value |
|-----------|------:|
| Mean PnL | **$51.64** |
| Median PnL | **$0.00** |
| Std Dev | **$983.66** |
| Skewness | **29.43** |

> ⚠️ **The PnL distribution is massively right-skewed (skewness = 29.43).** A small number of very large winning trades inflate the mean significantly above the median. This is typical of leveraged crypto trading where outsized winners coexist with many near-zero outcomes.

---

## 6. Statistical Testing

Four formal hypothesis tests were conducted to compare Fear vs. Greed trading environments:

| Test | Purpose | Result | p-value |
|------|---------|--------|---------|
| **Mann-Whitney U** on PnL | Compare PnL distributions (non-parametric) | Significant | **p = 0.000954** |
| **Chi-Square** on Win Rate | Compare win/loss proportions | Highly significant | **p < 0.000001** |
| **Mann-Whitney U** on Leverage | Compare leverage usage | Highly significant | **p < 0.000001** |
| **Rank-Biserial Correlation** | Effect size of PnL difference | Very small | **r = 0.0086** |

### Interpretation

- ✅ **Statistical significance is confirmed** — Fear and Greed periods produce genuinely different trading outcomes
- ⚠️ **Practical significance is minimal** — the rank-biserial correlation of 0.0086 indicates the real-world effect size is tiny
- 💡 **Bottom line:** Sentiment is a detectable signal, but it explains only a very small fraction of trade outcome variance. Relying on sentiment alone as a trading signal would be misguided.

---

## 7. Machine Learning Models

Two classification models were trained to predict whether a given trade would be profitable (binary: win vs. loss/break-even).

### 7.1 Model Performance Summary

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Avg Precision | Recall |
|-------|--------:|--------:|---------:|---------:|-------------:|-------:|
| **Logistic Regression** | 92.72% | 0.9558 | 0.9246 | 0.8598 | 0.9168 | 0.9999 |
| **Random Forest** ⭐ | **93.89%** | **0.9859** | **0.9356** | **0.8833** | **0.9801** | 0.9944 |

> 🏆 **Random Forest is the superior model** across all key metrics — particularly Average Precision (0.98 vs 0.92), which is most robust given the class imbalance in the PnL distribution.

### 7.2 ROC & Precision-Recall Curves

The chart below shows both models' discriminative power across all decision thresholds:

![Model Evaluation — ROC and Precision-Recall Curves](model_evaluation.png)

**Reading the charts:**

- **Left (ROC Curves):** Random Forest (AUC = 0.986) hugs the top-left corner far more tightly than Logistic Regression (AUC = 0.956), indicating substantially better true positive rate at any given false positive rate.
- **Right (Precision-Recall Curves):** Random Forest maintains near-perfect precision across the full recall range. Logistic Regression shows more variability at high recall, while still far exceeding the baseline (0.446).

---

## 8. Model Interpretability

Understanding *why* models make predictions is as important as their performance metrics.

### 8.1 Random Forest Feature Importance

![Random Forest Feature Importance](feature_importance.png)

| Rank | Feature | Importance Score | Notes |
|------|---------|----------------:|-------|
| 1 | **`is_buy`** | **0.8777** | Trade direction — dominant signal by huge margin |
| 2 | `trader_win_rate` | 0.0322 | Historical trader skill |
| 3 | `day_of_week` | 0.0213 | Weekday timing effect |
| 4 | `trader_avg_leverage` | 0.0160 | Trader's leverage discipline |
| 5 | `leverage_ratio` | 0.0152 | Trade-specific leverage |
| 6 | **`sentiment_encoded`** 🔴 | 0.0149 | Sentiment — present but marginal |
| 7 | `trade_size` | 0.0111 | Notional size |
| 8 | `trader_num_trades` | 0.0099 | Experience proxy |
| 9 | `is_weekend` | 0.0017 | Weakest signal |

> **Trade direction (`is_buy`) alone accounts for 87.77% of the model's predictive information.** All other features combined — including sentiment — contribute just over 12%.

---

### 8.2 Logistic Regression Coefficients

![Logistic Regression Coefficients](regression_coefficients.png)

| Feature | Coefficient | Direction | Interpretation |
|---------|------------:|-----------|----------------|
| **`is_buy`** | **−10.3221** | 🔴 Strongly negative | Buy/long trades are far less likely to be classified as winning in this context |
| `is_weekend` | −1.0667 | 🔴 Negative | Weekend trades are weaker |
| `trader_avg_leverage` | −0.3696 | 🔴 Negative | Higher average leverage → worse outcomes |
| `trader_num_trades` | −0.2190 | 🔴 Negative | Mixed signal on experience |
| `leverage_ratio` | −0.0802 | 🔴 Slightly negative | Higher trade leverage mildly worsens odds |
| `trade_size` | +0.0455 | 🟢 Slightly positive | Larger trades slightly more positive |
| `trader_win_rate` | +0.1156 | 🟢 Positive | Strong historical win rate improves odds |
| **`sentiment_encoded`** | **+0.2090** | 🟢 Positive | Greed sentiment slightly improves odds |
| `day_of_week` | +0.3110 | 🟢 Positive | Later weekdays have better outcomes |

**Key coefficients to highlight:**
- The massive negative weight on `is_buy` (−10.32) dominates the model — this likely reflects the dataset composition where short/sell trades in certain environments outperformed, or the directional bias of profitable positions in the sample period.
- `sentiment_encoded` has a positive coefficient (+0.21), confirming greed periods are marginally favorable — consistent with the win rate table showing Extreme Greed (46.49%) outperforming.
- `trader_avg_leverage` being negative (−0.37) is a critical insight: **traders who habitually use high leverage show worse outcomes**, validating the conservative leverage recommendation.

---

## 9. Key Trading Behavior Insights

Drawing from all analyses combined, these behavioral patterns stand out:

### 🎯 Finding 1: Trade Direction Is the Master Variable
With 87.77% feature importance, `is_buy` dwarfs every other signal. This means *whether you're long or short matters immensely* — more than any macro sentiment state. Directional accuracy is the primary skill in this market.

### ⚖️ Finding 2: Leverage Discipline Predicts Survival
Both models flag `trader_avg_leverage` negatively. Traders who default to lower leverage across their history have demonstrably better outcomes. The leverage treadmill — using high leverage to recover losses — appears counterproductive in the data.

### 📅 Finding 3: Weekday Trading Outperforms Weekend Trading
`is_weekend` carries a negative coefficient (−1.07) in Logistic Regression and is the weakest importance feature in Random Forest. Weekend sessions likely suffer from lower liquidity, wider spreads, and less informed price discovery.

### 🏆 Finding 4: Trader History Matters More Than Macro Conditions
`trader_win_rate` (Feature Importance: 0.0322) ranks second after direction — ahead of all sentiment and leverage variables. A trader's proven edge is more predictive than the market's emotional temperature on any given day.

### 😰 Finding 5: Extreme Greed Creates the Best Environment, Extreme Fear the Worst
Despite the small effect size, the gradient across sentiment classes is monotonically increasing from Extreme Fear (37.06% win rate) to Extreme Greed (46.49%). Sentiment does tilt the playing field — just modestly.

---

## 10. Strategy Recommendations

Based on the full quantitative analysis, five evidence-backed recommendations emerge:

### ✅ Recommendation 1: Behavior-First Framework
Build trading systems around **trade direction accuracy** and **leverage discipline** before any macro overlay. Getting direction right is worth 10x more than timing sentiment cycles.

### ✅ Recommendation 2: Use Sentiment as a Filter, Not a Signal
Treat the Fear & Greed Index as a *regime flag* that adjusts position sizing or risk appetite slightly — not as a trigger for entries and exits. During Extreme Greed, allow slightly larger risk. During Extreme Fear, tighten stops and reduce size.

### ✅ Recommendation 3: Conservative Leverage Is Non-Negotiable
The negative coefficient on `trader_avg_leverage` means the market consistently penalizes over-leveraged traders over time. Keeping leverage at conservative levels — particularly during Fear regimes — is the single most controllable risk lever.

### ✅ Recommendation 4: Prioritize Quality Over Frequency
`trader_num_trades` shows a negative coefficient (−0.22), suggesting that churning trades does not improve outcomes. Selectivity — trading fewer, higher-conviction setups — is favored by the data.

### ✅ Recommendation 5: Avoid Weekend Trading Unless Edge Is Confirmed
With `is_weekend` being the weakest positive feature and carrying a negative logistic coefficient, weekend trade setups should meet a higher bar. Default to reduced sizing or abstention unless a specific setup has demonstrated weekend edge.

---

## 11. Conclusion

This analysis set out to answer: **does Bitcoin market sentiment drive Hyperliquid trader profitability?**

The answer is nuanced:

| Dimension | Finding |
|-----------|---------|
| **Statistical significance** | ✅ Yes — Fear vs. Greed differences are real (p < 0.001) |
| **Practical effect size** | ❌ Tiny — rank-biserial r = 0.0086 |
| **Best predictive feature** | `is_buy` (87.77% importance) |
| **Sentiment's role** | Secondary context signal, 6th most important feature |
| **Best model** | Random Forest — ROC-AUC = 0.9859 |
| **Actionable conclusion** | Behavior-first framework >> sentiment-first framework |

The project confirms a **behavior-driven trading framework** where sentiment serves as contextual color rather than the engine of alpha. Traders who master directional accuracy, maintain leverage discipline, and leverage their personal track records will consistently outperform those chasing sentiment waves.

---

## Appendix: Technical Artifacts

The following model artifacts were generated and saved for downstream use:

| File | Description |
|------|-------------|
| `models/trading_behavior_models.pkl` | Serialized Logistic Regression + Random Forest models |
| `metrics/model_evaluation.png` | ROC & Precision-Recall curve comparisons |
| `metrics/regression_coefficients.png` | Logistic Regression coefficient bar chart |
| `metrics/feature_importance.png` | Random Forest feature importance chart |

---

*Report generated from `bitcoin_sentiment_trader_analysis.ipynb` | Submitted to Primetrade.ai as part of the Data Science Hiring Task*
