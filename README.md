# ML Portfolio Strategy Under Transaction Costs: Why Models Fail in Real Markets

> Built to understand not just how ML can be applied to financial markets — but more importantly, **why it fails.**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

- Live demo: *(add link — Streamlit Community Cloud / Hugging Face Spaces)*
- Screenshot: *(add `assets/dashboard.png` and link it here)*

## Key Findings

| Metric | ML Portfolio | Equal Weight |
|--------|-------------|--------------|
| Sharpe Ratio | 1.21 | 1.92 |
| Annual Turnover | 58.1x | — |
| Cost Drag | 5.81% | — |
| Annual Return | 31.54% | 40.52% |
| Max Drawdown | -16.48% | -9.53% |

> **Conclusion:** Transaction costs eliminate predictive edge in high-frequency rebalancing strategies — consistent with the Efficient Market Hypothesis.

---

## Why This Matters

This project is directly relevant to **portfolio managers, trading desks and risk teams** evaluating ML-driven strategies.

- High-frequency ML strategies fail due to transaction costs
- Risk-adjusted performance matters more than raw returns  
- Simple strategies can outperform complex models in efficient markets
- Understanding *why* models fail is more valuable than chasing better numbers

---

## What I Discovered

Honestly, the most interesting result wasn't what worked — it was what didn't. After adding realistic transaction costs, my ML model couldn't beat a simple equal-weight portfolio.

At first I thought something was wrong. Then I realised this is exactly what the Efficient Market Hypothesis predicts — and that understanding *why* the model fails is more valuable than chasing better numbers.

---

## Performance Degradation Pipeline

Raw Model → Fix Leakage → Add Costs → Weekly Rebalancing
Sharpe:  2.59  →  1.99  →  1.88  →  1.21

| Stage | Sharpe | What Changed |
|-------|--------|--------------|
| Raw model | 2.59 | Before any fixes |
| Fix data leakage | 1.99 | Proper feature shifting |
| Add transaction costs | 1.88 | 0.1% per trade, daily |
| Weekly rebalancing | 2.45 | Reduce turnover |
| Risk-adjusted weights | 1.21 | Final realistic result |
| Equal weight benchmark | 1.92 | Simple baseline |

---

## Why the ML Strategy Underperformed

**This is the most important part of the project.**

1. **High turnover (58x annually)** — daily rebalancing created 22.65% cost drag
2. **Momentum chasing** — model captured short-term noise, not stable signals
3. **Bull market bias** — Sharpe 2.51 in bull markets, -1.22 in bear markets
4. **Market efficiency** — after costs, simple equal-weight was more efficient

> Most people hide failure. This project investigates it.

---

## Regime Analysis

| Market | ML Sharpe | Equal Sharpe | Verdict |
|--------|-----------|--------------|---------|
| 🐂 Bull Market | 2.51 | 3.37 | Equal wins |
| 🐻 Bear Market | -1.22 | -1.08 | Equal wins |

The model is essentially a **momentum amplifier** — great in rising markets, damaging in falling ones.

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

## System Pipeline

Raw Prices → Log Returns → Feature Engineering → XGBoost Models
→ Risk-Adjusted Weights → Weekly Rebalancing → Transaction Costs
→ Performance Evaluation → Regime Analysis → Streamlit Dashboard

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

## Portfolio Construction

| Component | Details |
|-----------|---------|
| Weight signal | predicted_return / volatility |
| Max weight per asset | 20% |
| Rebalancing | Weekly (every 5 days) |
| Transaction cost | 0.1% per trade |
| Short selling | Not allowed |

---

## Transaction Cost Analysis

| Metric | Daily Rebalancing | Weekly Rebalancing |
|--------|------------------|-------------------|
| Annual turnover | ~250x | 58.1x |
| Annual cost drag | 22.65% | 5.81% |

> Switching to weekly rebalancing reduced cost drag by **75%** — the single biggest improvement.

---

## Key Learnings

1. **Data leakage is subtle** — rolling features must be shifted by 1 day to avoid using future information
2. **Transaction costs destroy alpha** — 22% drag from daily rebalancing exceeded all alpha generated
3. **Market efficiency is real** — after costs, equal weight beat ML consistently
4. **Regime matters** — model works in bull markets but amplifies losses in bear markets
5. **Turnover is the enemy** — reducing frequency was more impactful than improving the model

---

## Future Improvements

- Add sentiment analysis from financial news APIs
- Include macro features (VIX, interest rates, dollar index)
- Reduce turnover via position regularization constraints
- Test on 2008 financial crisis data for robustness
- Explore reinforcement learning for direct portfolio optimization
- Expand to 20+ assets for better diversification

---

## Project Structure

portfolio-m1/
├── notebooks/
│   ├── 01_data_and_returns.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 04_portfolio_construction.ipynb
├── data/
│   ├── prices.csv
│   ├── log_returns.csv
│   ├── volume.csv
├── app.py
├── requirements.txt
└── README.md

---

## How to Run

```bash
# Clone
git clone https://github.com/jathinrao07/ml-portfolio-optimization.git
cd portfolio-m1

# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Note: run `streamlit` from the repo root so `data/*.csv` paths resolve.

```

## Tests / CI

- Run locally: `pytest -q`
- CI: GitHub Actions workflow in `.github/workflows/ci.yml`

---

References
	•	Fama, E. (1970). Efficient Capital Markets
	•	Markowitz, H. (1952). Portfolio Selection
	•	Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System

About
I’m Jathin, an MSc Data Science student interested in quantitative finance and ML applications in financial markets. I built this project while preparing for UK finance internships.
The goal was not to build a model that works — but to understand why models fail under real-world constraints. That turned out to be the most valuable lesson.
🔗 GitHub: https://github.com/jathinrao07

Built April 2026 — feedback welcome