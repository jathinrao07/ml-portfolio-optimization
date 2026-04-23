# ML Portfolio Strategy with Realistic Backtesting

> A critical evaluation of ML-driven portfolio strategies under real market constraints —
> including transaction costs, regime analysis, and market efficiency testing.

## Key Results

| Metric | ML Portfolio | Equal Weight |
|--------|-------------|--------------|
| Sharpe Ratio | 1.21 | 1.92 |
| Annual Return | 31.54% | 40.52% |
| Max Drawdown | -16.48% | -9.53% |
| Annual Turnover | 58.1x | — |
| Cost Drag | 5.81% | — |

**Bottom line:** ML strategy underperformed after realistic constraints — and that's the most valuable finding.

---

## Core Insight

While the ML model showed initial predictive power (raw Sharpe 2.59), performance degraded significantly once realistic constraints were applied:
Raw Model → Fix Leakage → Add Costs → Weekly Rebalancing
Sharpe: 2.59 → 1.99 → 1.88 → 1.21
This highlights the difficulty of generating consistent alpha in efficient markets — consistent with Fama's Efficient Market Hypothesis.

---

## Why the ML Strategy Underperformed

This is the most important part of the project.

1. **High turnover (58x annually)** — daily rebalancing created 22.65% cost drag
2. **Momentum chasing** — model captured short-term noise, not stable signals
3. **Bull market bias** — Sharpe 2.51 in bull markets, -1.22 in bear markets
4. **Market efficiency** — after costs, simple equal-weight was more efficient

> Most people hide failure. This project investigates it.

---

## What Makes This Different

Compared to typical ML portfolio projects on GitHub:

| Feature | Typical Projects | This Project |
|---------|-----------------|--------------|
| Transaction costs | ❌ Ignored | ✅ 0.1% per trade modeled |
| Data leakage fix | ❌ Often present | ✅ Features properly shifted |
| Regime analysis | ❌ Rare | ✅ Bull/bear performance split |
| Turnover analysis | ❌ Not included | ✅ 58.1x annual turnover measured |
| Honest conclusions | ❌ Cherry-picked | ✅ Model failure documented |

---

## Pipeline
Raw Prices → Log Returns → Feature Engineering → XGBoost Models
→ Risk-Adjusted Weights → Weekly Rebalancing → Transaction Costs
→ Performance Evaluation → Regime Analysis → Dashboard
---

## Features (19 per asset)

| Category | Features |
|----------|---------|
| Momentum | Rolling mean returns (5, 10 days) |
| Volatility | Rolling std deviation (10, 20 days) |
| Trend | MA10, MA50, MA ratio |
| Technical | RSI, MACD, MACD signal |
| Volume | Volume, volume MA, volume ratio |
| Cross-asset | SPY return, SPY momentum, SPY correlation |
| Calendar | Day of week |

---

## Regime Analysis

| Market | ML Sharpe | Equal Sharpe | Verdict |
|--------|-----------|--------------|---------|
| Bull Market | 2.51 | 3.37 | Equal wins |
| Bear Market | -1.22 | -1.08 | Equal wins |

The model is essentially a momentum amplifier — great in rising markets, damaging in falling ones.

---

## Performance Degradation Pipeline

| Stage | Sharpe | What Changed |
|-------|--------|--------------|
| Raw model | 2.59 | Before any fixes |
| Fix data leakage | 1.99 | Proper feature shifting |
| Add transaction costs | 1.88 | 0.1% per trade, daily |
| Weekly rebalancing | 2.45 | Reduce turnover |
| Risk-adjusted weights | 1.21 | Final realistic result |
| Equal weight benchmark | 1.92 | Simple baseline |

---

## How to Run

```bash
# Clone
git clone https://github.com/jathinrao07/ml-portfolio-optimization.git
cd ml-portfolio-optimization

# Install
pip install yfinance pandas numpy scikit-learn xgboost ta streamlit matplotlib seaborn

# Run notebooks in order
jupyter lab
# 01 → 02 → 03 → 04

# Launch dashboard
streamlit run app.py
portfolio-m1/
├── notebooks/
│   ├── 01_data_and_returns.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_portfolio_construction.ipynb
├── data/
│   ├── prices.csv
│   ├── log_returns.csv
│   ├── all_features.pkl
│   ├── models.pkl
│   └── predictions.pkl
├── app.py
└── README.md
Key Learnings
	1.	Data leakage is subtle — rolling features must be shifted by 1 day
	2.	Transaction costs destroy alpha — 22% drag from daily rebalancing
	3.	Market efficiency is real — equal weight beat ML after costs
	4.	Regime matters — model works in bull markets, fails in bear markets
	5.	Turnover is the enemy — reducing from daily to weekly rebalancing was the single biggest improvement
References
	•	Fama, E. (1970). Efficient Capital Markets
	•	Markowitz, H. (1952). Portfolio Selection
	•	Chen & Guestrin (2016). XGBoost
About
I’m Jathin, an MSc Data Science student interested in quantitative finance.
I built this project while preparing for UK finance internships — the goal
was not to build a model that works, but to understand why models fail.
🔗 GitHub
Built April 2026 — feedback welcome
