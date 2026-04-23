# 🤖 Multi-Asset ML Portfolio Optimization System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An end-to-end quantitative finance system that uses machine learning to predict 
> asset returns and construct dynamic portfolios — tested with real market data, 
> realistic transaction costs, and institutional-grade risk metrics.

---

## 🎯 Project Overview

This project builds a complete ML-driven portfolio management system:

1. **Predict** next-day log returns for 6 assets using XGBoost
2. **Construct** dynamic portfolios using risk-adjusted weights
3. **Evaluate** performance against benchmarks with institutional metrics
4. **Analyse** why the model fails — regime analysis, transaction costs, market efficiency

### Key Finding
> *"After incorporating transaction costs and realistic rebalancing, the ML strategy 
> struggled to consistently outperform a simple equal-weight benchmark — consistent 
> with the Efficient Market Hypothesis and the difficulty of extracting stable alpha 
> in modern financial markets."*

---

## 📊 Results

| Metric | ML Portfolio | Equal Weight |
|--------|-------------|--------------|
| Sharpe Ratio | 1.21 | **1.92** |
| Annual Return | 31.54% | **40.52%** |
| Max Drawdown | -16.48% | **-9.53%** |
| Daily Volatility | 1.38% | **1.17%** |

### Regime Analysis
| Market | ML Sharpe | Equal Sharpe |
|--------|-----------|--------------|
| 🐂 Bull Market | 2.51 | 3.37 |
| 🐻 Bear Market | -1.22 | -1.08 |

---

## 🧱 Project Structure
portfolio-m1/
│
├── notebooks/
│   ├── 01_data_and_returns.ipynb│   ├── 02_feature_engineering.ipynb│   ├── 03_model_training.ipynb│   └── 04_portfolio_construction.ipynb│
├── data/
│   ├── prices.csv│   ├── log_returns.csv│   ├── volume.csv│   ├── all_features.pkl│   ├── models.pkl│   └── predictions.pkl│
├── app.py└── README.md

---

## ⚙️ Feature Engineering (19 features per asset)

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

## 💰 Portfolio Construction

| Component | Details |
|-----------|---------|
| Weight signal | predicted_return / volatility |
| Max weight | 20% per asset |
| Rebalancing | Weekly (every 5 days) |
| Transaction cost | 0.1% per trade |
| Short selling | Not allowed |

---

## 📉 Transaction Cost Analysis

| Metric | Daily | Weekly |
|--------|-------|--------|
| Annual turnover | ~250x | 58.1x |
| Annual cost drag | 22.65% | **5.81%** |

> Switching to weekly rebalancing reduced cost drag by **75%**

---

## 🧠 Key Learnings

**1. Data Leakage is subtle**
Rolling features must be shifted by 1 day to avoid using future information.

**2. Transaction costs destroy alpha**
Daily rebalancing created 22% annual cost drag — more than the alpha generated.

**3. Market efficiency is real**
After costs, simple equal-weight outperformed ML — consistent with EMH.

**4. Regime matters**
Model captures momentum in bull markets but amplifies losses in bear markets.

---

## 🔧 Improvements Made

| Improvement | Impact |
|-------------|--------|
| Fixed data leakage | More reliable results |
| Risk-adjusted weights | Sharpe 1.88 → 1.99 |
| Weekly rebalancing | Cost drag 22% → 5.81% |
| XGBoost tuning | RMSE improved all assets |

---

## 🚀 Future Work

- Add sentiment analysis from financial news
- Include macro features (VIX, interest rates, dollar index)
- Test on 2008 financial crisis data
- Implement walk-forward validation
- Expand to 20+ assets

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/jathinrao07/ml-portfolio-optimization.git
cd ml-portfolio-optimization

# Install dependencies
pip install yfinance pandas numpy scikit-learn xgboost ta streamlit matplotlib seaborn

# Run notebooks in order (1 → 4)
jupyter lab

# Launch dashboard
streamlit run app.py

📚 References
	•	Fama, E. (1970). Efficient Capital Markets
	•	Markowitz, H. (1952). Portfolio Selection
	•	Chen & Guestrin (2016). XGBoost

👤 Author
Kadaru Jathin MSc Data Science🔗 GitHub

Built as part of a quantitative finance internship project — April 2026
