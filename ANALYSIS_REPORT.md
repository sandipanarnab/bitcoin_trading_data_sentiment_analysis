# Bitcoin Market Sentiment and Trader Performance Analysis
## Hyperliquid Trading Insights: Fear vs Greed

## Executive Summary

This project investigates whether Bitcoin market sentiment, measured by the Fear and Greed Index, is associated with trader performance on Hyperliquid. The notebook merges daily sentiment data with trade-level execution data, engineers trader and trade behavior features, and evaluates both statistical and machine learning models.

The main conclusion is clear: sentiment has a statistically detectable but practically small relationship with performance. Trade direction, leverage discipline, and trader-specific behavior are much stronger signals than sentiment alone.

### Key Numbers

| Metric | Value |
| --- | ---: |
| Fear and Greed observations | 2,644 |
| Raw Hyperliquid trades | 211,224 |
| Final merged trades used for analysis | 173,532 |
| Unique traders | 32 |
| Best model | Random Forest |
| Best ROC-AUC | 0.9859 |

## Data Sources

### 1. Bitcoin Market Sentiment Dataset
- Daily Fear and Greed Index values
- Coverage: 2018-02-01 to 2025-05-02
- Fields used: timestamp, value, classification, date

### 2. Historical Hyperliquid Trader Data
- Trade-level execution records
- Coverage used in notebook: 2023-05-01 to 2025-04-30
- Fields used: account, coin, execution price, size, side, start position, direction, closed PnL, leverage, timestamps

### 3. Project Documents
- `project_document/DS Task.docx` confirms the assignment goal: analyze how market sentiment relates to trader behavior and profitability
- `project_document/Explanation.pdf` provides the same challenge context in PDF form

## Methodology

The notebook follows a full analysis pipeline:

1. Load sentiment and trade datasets
2. Clean date fields and standardize categorical values
3. Merge trades to daily sentiment labels
4. Remove neutral sentiment rows to isolate Fear vs Greed comparisons
5. Engineer features such as:
   - trade size
   - buy/sell indicator
   - trader historical win rate
   - average leverage
   - day of week and weekend flags
6. Run exploratory analysis and hypothesis tests
7. Train Logistic Regression and Random Forest models
8. Interpret coefficients and feature importance
9. Save model artifacts for reuse

## Data Preparation Results

### Merge Coverage
- Trades with sentiment labels: 211,218
- Trades without sentiment labels: 6
- Missing sentiment coverage: 0.0028%
- Trades remaining after removing neutral sentiment and incomplete rows: 173,532

### Sentiment Distribution

| Classification | Trades |
| --- | ---: |
| Fear | 61,837 |
| Greed | 50,303 |
| Extreme Greed | 39,992 |
| Extreme Fear | 21,400 |

### Binary Sentiment Split

| Label | Trades |
| --- | ---: |
| Fear | 83,237 |
| Greed | 90,295 |

## Exploratory Findings

### Sentiment and Win Rate

Binary comparison:

| Sentiment | Mean PnL | Median PnL | Win Rate |
| --- | ---: | ---: | ---: |
| Fear | 49.2121 | 0.0000 | 40.79% |
| Greed | 53.8823 | 0.0000 | 42.03% |

By sentiment class:

| Classification | Mean PnL | Median PnL | Win Rate |
| --- | ---: | ---: | ---: |
| Extreme Fear | 34.5379 | 0.0000 | 37.06% |
| Fear | 54.2904 | 0.0000 | 42.08% |
| Greed | 42.7436 | 0.0000 | 38.48% |
| Extreme Greed | 67.8929 | 0.0000 | 46.49% |

### PnL Shape
- Mean PnL: $51.64
- Median PnL: $0.00
- Standard deviation: $983.66
- Skewness: 29.43
- Interpretation: the distribution is highly right-skewed, so a few large gains dominate the average

## Statistical Tests

The notebook runs several hypothesis tests to compare Fear and Greed regimes.

| Test | Result |
| --- | --- |
| Mann-Whitney U test on PnL | p = 0.000954 |
| Chi-square test on win rate | p < 0.000001 |
| Mann-Whitney U test on leverage | p < 0.000001 |
| Rank-biserial correlation | 0.0086 |

### Interpretation
- Sentiment differences are statistically significant
- The practical effect size is very small
- That means sentiment is measurable, but not the main driver of profitability

## Model Performance

### Evaluation Table

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Avg Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9272 | 0.9558 | 0.9246 | 0.8598 | 0.9168 | 0.9999 |
| Random Forest | 0.9389 | 0.9859 | 0.9356 | 0.8833 | 0.9801 | 0.9944 |

### Model Takeaway
- Random Forest is the best performer overall
- Logistic Regression remains useful for coefficient interpretation
- Both models show that behavior-related variables carry most of the signal

## Interpretability

### Logistic Regression Coefficients

| Feature | Coefficient |
| --- | ---: |
| is_buy | -10.3221 |
| is_weekend | -1.0667 |
| trader_avg_leverage | -0.3696 |
| trader_num_trades | -0.2190 |
| leverage_ratio | -0.0802 |
| trade_size | 0.0455 |
| trader_win_rate | 0.1156 |
| sentiment_encoded | 0.2090 |
| day_of_week | 0.3110 |

### Random Forest Feature Importance

| Feature | Importance |
| --- | ---: |
| is_buy | 0.8777 |
| trader_win_rate | 0.0322 |
| day_of_week | 0.0213 |
| trader_avg_leverage | 0.0160 |
| leverage_ratio | 0.0152 |
| sentiment_encoded | 0.0149 |
| trade_size | 0.0111 |
| trader_num_trades | 0.0099 |
| is_weekend | 0.0017 |

### Interpretation
- Trade direction is the strongest signal by a wide margin
- Trader-specific history matters more than sentiment
- Sentiment features do help, but only marginally

## Trading Behavior Insights

- Lower leverage is associated with better outcomes
- Trader-specific behavior dominates macro sentiment in predictive value
- Weekend trading is weaker than weekday trading in the notebook summary
- Trade frequency shows mixed signals, so it should not be treated as the primary edge

## Strategy Recommendations

1. Keep leverage conservative, especially during weaker sentiment periods
2. Prioritize trade quality over raw trade count
3. Treat sentiment as a secondary filter, not a standalone signal
4. Use trader history and directionality as the main inputs to decision-making
5. Reserve larger risk for setups that align with both sentiment and trader profile

## Generated Artifacts

The notebook produces the following outputs:

- `metrics/model_evaluation.png`
- `metrics/regression_coefficients.png`
- `metrics/feature_importance.png`
- `models/trading_behavior_models.pkl`

## Conclusion

The analysis shows that Bitcoin sentiment has a real but limited effect on Hyperliquid trader performance. The strongest predictors are trade direction and trader behavior, especially leverage discipline and historical win rate. The project therefore supports a behavior-first trading framework, where sentiment acts as a context signal rather than the main driver of profitability.

