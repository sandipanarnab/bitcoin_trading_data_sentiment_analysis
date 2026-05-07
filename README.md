# 🪙 Bitcoin Market Sentiment & Trader Performance Analysis

> **Submitted to:** Primetrade.ai Hiring Team — Data Science Assignment  
> **Platform Analyzed:** Hyperliquid Perpetuals Exchange  
> **Coverage:** February 2018 – May 2025

Analyze how Bitcoin market sentiment affects trader behavior and performance using Hyperliquid trade data.

📄 **For the full written analysis, findings, and strategy recommendations, see [`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md)**

## What This Project Does

This repository contains a notebook-driven analysis that:

- merges Bitcoin Fear and Greed sentiment with Hyperliquid trade records
- engineers trader-level and trade-level features
- compares Fear vs. Greed regimes statistically
- trains Logistic Regression and Random Forest models
- saves plots and model artifacts for reuse

## Main Findings

- Sentiment differences between Fear and Greed are **statistically significant** (Mann-Whitney p = 0.000954), but the practical effect size is very small (rank-biserial r = 0.0086)
- **Random Forest** is the best model with a ROC-AUC of **0.9859** and accuracy of **93.89%**
- **Trade direction (`is_buy`) is the strongest predictive feature by far**, accounting for 87.77% of Random Forest importance
- Trader history (win rate) and leverage discipline matter more than sentiment alone
- **Extreme Greed** produces the highest win rate (46.49%) and mean PnL ($67.89); **Extreme Fear** the lowest (37.06%)
- Weekend trading consistently underperforms weekday trading across both models

> 📄 See [`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md) for the full breakdown with charts, statistical tests, and trading strategy recommendations.

## Repository Structure

```text
project_document/              Assignment brief and explanation files
notebook/bitcoin_sentiment_trader_analysis.ipynb
metrics/                       Saved plots from the notebook
models/                        Serialized model bundle
ANALYSIS_REPORT.md             ← Full written analysis report (start here)
requirements.txt               Python dependencies
```

## Key Outputs

| Output | Description |
| --- | --- |
| **[`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md)** | **📄 Full written analysis — findings, charts, and strategy recommendations** |
| `metrics/model_evaluation.png` | ROC and precision-recall curves for both models |
| `metrics/regression_coefficients.png` | Logistic Regression coefficient chart |
| `metrics/feature_importance.png` | Random Forest feature importance chart |
| `models/trading_behavior_models.pkl` | Saved model bundle and metadata |

## Notebook Summary

- Fear and Greed dataset: 2,644 daily observations
- Raw Hyperliquid trades: 211,224
- Final merged analysis set: 173,532 trades
- Unique traders: 32
- Best model: Random Forest
- Best ROC-AUC: 0.9859

## Key Metrics

### Binary Sentiment Comparison

| Sentiment | Win Rate | Mean PnL |
| --- | ---: | ---: |
| Fear | 40.79% | 49.2121 |
| Greed | 42.03% | 53.8823 |

### Model Evaluation

| Model | Accuracy | ROC-AUC | F1-Score |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.9272 | 0.9558 | 0.9246 |
| Random Forest | 0.9389 | 0.9859 | 0.9356 |

## Getting Started

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Open the notebook

Open `notebook/bitcoin_sentiment_trader_analysis.ipynb` in Jupyter, VS Code, or your preferred notebook environment and run the cells from top to bottom.

## Inputs

- `data/original/fear_greed_index.csv`
- `data/original/historical_data.csv`

## Notes

- The notebook saves artifacts into `metrics/` and `models/`
- The analysis report in `ANALYSIS_REPORT.md` provides the full written summary
- The assignment context is stored in `project_document/`

## Limitations

- Sentiment is useful, but it is not the dominant signal — the rank-biserial correlation of 0.0086 shows a tiny practical effect size despite statistical significance
- The strongest model feature is trade direction, so the analysis is more behavioral than macro-driven
- Results for trade frequency are mixed in the notebook — leverage discipline and directional accuracy are more reliable decision variables than activity count alone
- Only 32 unique traders in the dataset; broader generalization would benefit from a larger trader pool

## Further Reading

The [`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md) file contains the complete analysis including:
- Full EDA with PnL distribution breakdown across all 4 sentiment classes
- Statistical test results (Mann-Whitney U, Chi-Square, effect sizes)
- Model performance deep-dive with ROC and Precision-Recall curve interpretation
- Feature importance and Logistic Regression coefficient analysis
- 5 actionable strategy recommendations grounded in model outputs

