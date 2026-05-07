# bitcoin_trading_data_sentiment_analysis

Analyze how Bitcoin market sentiment affects trader behavior and performance using Hyperliquid trade data.

## What This Project Does

This repository contains a notebook-driven analysis that:

- merges Bitcoin Fear and Greed sentiment with Hyperliquid trade records
- engineers trader-level and trade-level features
- compares Fear vs Greed regimes statistically
- trains Logistic Regression and Random Forest models
- saves plots and model artifacts for reuse

## Main Findings

- Sentiment differences are statistically significant, but the practical effect is small
- Random Forest is the best model, with ROC-AUC of 0.9859
- Trade direction is the strongest predictive feature by far
- Trader history and leverage discipline matter more than sentiment alone

## Repository Structure

```text
project_document/              Assignment brief and explanation files
notebook/bitcoin_sentiment_trader_analysis.ipynb
metrics/                       Saved plots from the notebook
models/                        Serialized model bundle
ANALYSIS_REPORT.md             Full written analysis
requirements.txt               Python dependencies
```

## Key Outputs

| Output | Description |
| --- | --- |
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

- Sentiment is useful, but it is not the dominant signal
- The strongest model feature is trade direction, so the analysis is more behavioral than macro-driven
- Some results in the notebook are mixed for trade frequency, so leverage and direction are better decision variables than activity count alone

