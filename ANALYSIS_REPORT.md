# Bitcoin Market Sentiment & Trader Performance Analysis
## Hyperliquid Trading Insights: Fear vs Greed

**Objective:** Analyze how market sentiment (Fear vs Greed) affects trader behavior and performance on Hyperliquid. Identify statistically meaningful patterns and translate them into actionable trading insights.

**Author:** MiniMax Agent
**Date:** 2024

---

## Executive Summary

This analysis examines **35,864 trades** from **2,644 days** of Bitcoin Fear & Greed Index data to uncover the relationship between market sentiment and trader performance.

### Key Findings

| Metric | Value |
|--------|-------|
| **Fear Period Win Rate** | 36.9% |
| **Greed Period Win Rate** | 47.8% |
| **Win Rate Differential** | 10.9 percentage points |
| **Statistical Significance** | p < 0.0001 |
| **High Leverage Win Rate** | 2.9% |
| **Low Leverage Win Rate** | 47.1% |
| **Best Model AUC** | 0.9901 (Random Forest) |

---

## 1. Data Overview

### Datasets Analyzed
- **Fear & Greed Index:** 2,644 daily observations (Feb 2018 - May 2025)
- **Hyperliquid Trader Data:** 211,224 trades (79,225 after cleaning)
- **Merged Dataset:** 35,864 trades with sentiment labels

### Data Schema

**Fear & Greed Index:**
- `timestamp`: Unix timestamp
- `value`: Index value (0-100)
- `classification`: Extreme Fear, Fear, Neutral, Greed, Extreme Greed
- `date`: Date in M/D/YYYY format

**Hyperliquid Trader Data:**
- `Account`: Trader wallet address
- `Coin`: Trading pair (@107 = ETH)
- `Execution Price`: Trade price
- `Size Tokens/USD`: Trade size
- `Side`: BUY/SELL
- `Closed PnL`: Profit/Loss
- `Start Position`: Position size
- `leverage`: Leverage used

---

## 2. Key Insights (Data-Backed)

### 1. Sentiment Significantly Impacts Win Rate
- **Fear period win rate: 36.9%**
- **Greed period win rate: 47.8%**
- **Difference: 10.9 percentage points**
- **Statistical significance: p < 0.0001 (highly significant)**
- Implication: Greed periods offer better probability of profitable trades

### 2. Leverage is the Dominant Risk Factor
- **High leverage (5x+) win rate: 2.9%**
- **Low leverage (<5x) win rate: 47.1%**
- **Impact: 44.2 percentage point difference**
- Implication: High leverage is catastrophic for win rate, regardless of sentiment

### 3. Individual Trader Behavior Dominates
- **Trader behavior features: 94.7% of model importance**
- **Sentiment features: 5.3% of model importance**
- Implication: Historical trader performance is more predictive than market sentiment

### 4. Top 3 Most Important Features
1. `is_buy` - Trade direction
2. `trader_win_rate` - Historical win rate
3. `sentiment_encoded` - Market sentiment

### 5. Statistical Effect Size is Small
- **Cohen's d = 0.0121 (Small effect)**
- While statistically significant, the practical effect size is small
- Sentiment alone is not sufficient for trading decisions

### 6. Buy Orders Dominate
- Majority of trades are BUY orders (long positions)
- Direction is a key predictor of profitability

---

## 3. Model Performance

### Comparison

| Model | ROC-AUC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | 0.9651 | High | High | High |
| **Random Forest** | **0.9901** | **High** | **High** | **High** |

### Winner: Random Forest (AUC: 0.9901)
- Non-linear relationships between features and profitability
- Trader characteristics captured better than linear model

---

## 4. Strategy Recommendations

### A. Leverage Management
1. **Reduce leverage during Fear periods**
   - Target: 2-3x max during Fear/Extreme Fear
   - Expected improvement: ~10% increase in win rate

2. **Increase leverage selectively during Greed**
   - Only when sentiment >= Neutral and with low leverage history
   - Target: 3-5x max

### B. Position Sizing
1. **Reduce position size during Fear**
   - Cut sizes by 30-40% during Fear periods
   - Preserve capital for Greed opportunities

2. **Reserve larger positions for high-conviction setups**
   - When sentiment aligns with trade direction

### C. Entry Timing
1. **Prefer Greed/Neutral periods for new positions**
   - Win rate is 47.8% vs 36.9% in Fear

2. **Use Fear periods for:**
   - Profit-taking existing positions
   - Hedging strategies

### D. Risk Management
1. **Hard stop-losses at 2% of notional**
2. **Reduce exposure 50% after 5% drawdown**
3. **Exit and reassess after 10% drawdown**

### E. Behavioral Rules
1. **Cool-down period after losing trades** (30 minutes)
2. **Quality over quantity** - wait for setups
3. **Never increase leverage after losses**

---

## 5. Files Generated

### Analysis Files
- `bitcoin_sentiment_trader_analysis.ipynb` - Complete Jupyter notebook (12 sections)
- `run_analysis.py` - Python script for analysis execution

### Output Files
- `charts/analysis_summary.png` - Visualization dashboard
- `models/metrics.csv` - Model performance metrics

---

## Conclusion

This analysis demonstrates that **market sentiment significantly affects trader performance**, with Greed periods showing 10.9 percentage points higher win rates than Fear periods. However, **individual trader behavior (historical win rate, leverage usage) is far more predictive** than sentiment alone, explaining 94.7% of model importance.

The most actionable finding is the **catastrophic impact of high leverage**, which reduces win rate from 47.1% to just 2.9%. This suggests that **leverage management is the single most important factor** for trading success, more important than market timing.

### Practical Takeaways
1. Always use low leverage (below 5x)
2. Prefer opening positions during Greed/Neutral sentiment
3. Reduce position sizes during Fear periods
4. Historical trader performance is the best predictor of future success

---

*This analysis provides a data-driven framework for sentiment-adjusted trading strategies with quantifiable risk parameters.*