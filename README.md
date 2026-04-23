# ML Portfolio Strategy Under Transaction Costs: Why Models Fail in Real Markets

A critical evaluation of ML-driven portfolio strategies under real market constraints — including transaction costs, regime analysis, and market efficiency testing.

## Key Findings

| Metric | ML Portfolio | Equal Weight |
|--------|-------------|--------------|
| Sharpe Ratio | 1.21 | 1.92 |
| Annual Turnover | 58.1x | — |
| Cost Drag | 5.81% | — |
| Annual Return | 31.54% | 40.52% |

**Conclusion:** Transaction costs eliminate predictive edge in high-frequency rebalancing strategies — consistent with the Efficient Market Hypothesis.

## Why This Matters

- High-frequency ML strategies fail due to transaction costs
- Risk-adjusted performance matters more than raw returns
- Simple strategies can outperform complex models in efficient markets
- Understanding why models fail is more valuable than chasing better numbers

## Performance Degradation Pipeline

| Stage | Sharpe | What Changed |
|-------|--------|--------------|
| Raw model | 2.59 | Before any fixes |
| Fix data leakage | 1.99 | Proper feature shifting |
| Add transaction costs | 1.88 | 0.1% per trade daily |
| Weekly rebalancing | 2.45 | Reduce turnover |
| Final result | 1.21 | Risk-adjusted weights |
| Equal weight | 1.92 | Simple baseline |

## Regime Analysis

| Market | ML Sharpe | Equal Sharpe | Verdict |
|--------|-----------|--------------|---------|
| Bull Market | 2.51 | 3.37 | Equal wins |
| Bear Market | -1.22 | -1.08 | Equal wins |

The model is a momentum amplifier — great in rising markets, damaging in falling ones.

## Why the ML Strategy Underperformed

1. **High turnover (58x annually)** — daily rebalancing created 22.65% cost drag
2. **Momentum chasing** — model captured short-term noise, not stable signals
3. **Bull market bias** — Sharpe 2.51 in bull markets, -1.22 in bear markets
4. **Market efficiency** — after costs, simple equal-weight was more efficient

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

## How to Run

git clone https://github.com/jathinrao07/ml-portfolio-optimization.git
cd ml-portfolio-optimization
pip install yfinance pandas numpy scikit-learn xgboost ta streamlit matplotlib seaborn
jupyter lab
streamlit run app.py

## Key Learnings

1. Data leakage is subtle — rolling features must be shifted by 1 day
2. Transaction costs destroy alpha — 22% drag from daily rebalancing
3. Market efficiency is real — equal weight beat ML after costs
4. Regime matters — model works in bull markets, fails in bear markets

## Future Improvements

- Incorporate macro features (VIX, interest rates, dollar index)
- Reduce turnover via position regularization
- Test on 2008 financial crisis data for robustness
- Explore reinforcement learning for direct portfolio optimization
- Expand to 20+ assets for better diversification

## References

- Fama, E. (1970). Efficient Capital Markets
- Markowitz, H. (1952). Portfolio Selection
- Chen & Guestrin (2016). XGBoost

## About

I'm Jathin, an MSc Data Science student interested in quantitative finance.
I built this project while preparing for UK finance internships — the goal
was not to build a model that works, but to understand why models fail.

Built April 2026 — feedback welcome