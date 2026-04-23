import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    with open("data/predictions.pkl", "rb") as f:
        predictions = pickle.load(f)
    with open("data/all_features.pkl", "rb") as f:
        all_features = pickle.load(f)
    return predictions, all_features

ASSETS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'SPY', 'GLD']
predictions, all_features = load_data()

# Rebuild returns
pred_returns = pd.DataFrame()
actual_returns = pd.DataFrame()
for asset in ASSETS:
    pred_returns[asset] = predictions[asset]['y_pred']
    actual_returns[asset] = predictions[asset]['y_test'].values

# Risk adjusted weights
def get_ml_weights(pred_returns, all_features, max_weight=0.20):
    weights = pd.DataFrame(index=pred_returns.index, columns=pred_returns.columns)
    for i in range(len(pred_returns)):
        row = pred_returns.iloc[i]
        vol = pd.Series(index=ASSETS, dtype=float)
        for asset in ASSETS:
            feat = all_features[asset]
            if i < len(feat):
                vol[asset] = feat['volatility_10'].iloc[i]
            else:
                vol[asset] = 1.0
        vol = vol.replace(0, vol.mean()).fillna(vol.mean())
        signal = row / vol
        positive = signal[signal > 0]
        if len(positive) == 0:
            w = pd.Series(1/len(ASSETS), index=ASSETS)
        else:
            w = positive / positive.sum()
            w = w.clip(upper=max_weight)
            w = w / w.sum()
            full_w = pd.Series(0.0, index=ASSETS)
            full_w[w.index] = w
            w = full_w
        weights.iloc[i] = w
    return weights.astype(float)

ml_weights = get_ml_weights(pred_returns, all_features)

# Weekly rebalancing
ml_weights_weekly = ml_weights.copy()
for i in range(len(ml_weights)):
    if i % 5 != 0:
        ml_weights_weekly.iloc[i] = ml_weights_weekly.iloc[i-1]

# Portfolio returns with transaction costs
TRANSACTION_COST = 0.001
weight_changes = ml_weights_weekly.diff().abs().sum(axis=1)
daily_costs = weight_changes * TRANSACTION_COST
ml_portfolio_returns = pd.Series((ml_weights_weekly.values * actual_returns.values).sum(axis=1))
ml_portfolio_returns_after_costs = ml_portfolio_returns - daily_costs
equal_portfolio_returns = pd.Series((np.ones((len(actual_returns), len(ASSETS))) / len(ASSETS) * actual_returns.values).sum(axis=1))

# Cumulative returns
ml_cumulative = (1 + ml_portfolio_returns_after_costs).cumprod()
equal_cumulative = (1 + equal_portfolio_returns).cumprod()

# Regime analysis
spy_returns = actual_returns['SPY']
spy_rolling = spy_returns.rolling(20).mean()
regime = pd.Series('Bull', index=spy_returns.index)
regime[spy_rolling < 0] = 'Bear'
bull_days = regime == 'Bull'
bear_days = regime == 'Bear'

# Metrics
rf = 0.05 / 252
def get_metrics(returns):
    sharpe = (returns.mean() - rf) / returns.std() * np.sqrt(252)
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    return {
        'Sharpe Ratio': round(sharpe, 2),
        'Annual Return': f"{round(returns.mean() * 252 * 100, 2)}%",
        'Max Drawdown': f"{round(drawdown.min() * 100, 2)}%",
        'Daily Volatility': f"{round(returns.std() * 100, 2)}%"
    }

def get_sharpe(returns):
    return round((returns.mean() - rf) / returns.std() * np.sqrt(252), 2)

# Dashboard
st.title("Multi-Asset ML Portfolio Dashboard")
st.markdown("*XGBoost | Risk-adjusted weights | Weekly rebalancing | 0.1% transaction costs*")
st.markdown("---")

# Metrics
st.subheader("Overall Portfolio Metrics")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**ML Portfolio**")
    for k, v in get_metrics(ml_portfolio_returns_after_costs).items():
        st.metric(k, v)
with col2:
    st.markdown("**Equal Weight Benchmark**")
    for k, v in get_metrics(equal_portfolio_returns).items():
        st.metric(k, v)

st.markdown("---")

# Equity curve
st.subheader("Cumulative Returns")
fig, ax = plt.subplots(figsize=(12, 5))
ml_cumulative.plot(ax=ax, label='ML Portfolio', color='blue', linewidth=2)
equal_cumulative.plot(ax=ax, label='Equal Weight', color='orange', linewidth=2)
ax.legend()
ax.grid(True)
ax.set_xlabel("Days")
ax.set_ylabel("Portfolio Value ($1 invested)")
st.pyplot(fig)

st.markdown("---")

# Regime analysis
st.subheader("Regime Analysis — When Does the Model Work?")
col1, col2 = st.columns(2)

ml_returns = ml_portfolio_returns_after_costs.values
equal_returns = equal_portfolio_returns.values

with col1:
    st.markdown("**Bull Market**")
    st.metric("ML Sharpe", get_sharpe(pd.Series(ml_returns[bull_days.values])))
    st.metric("Equal Sharpe", get_sharpe(pd.Series(equal_returns[bull_days.values])))

with col2:
    st.markdown("**Bear Market**")
    st.metric("ML Sharpe", get_sharpe(pd.Series(ml_returns[bear_days.values])))
    st.metric("Equal Sharpe", get_sharpe(pd.Series(equal_returns[bear_days.values])))

fig2, ax2 = plt.subplots(figsize=(12, 4))
data = {
    'ML Bull': pd.Series(ml_returns[bull_days.values]).mean() * 252 * 100,
    'Equal Bull': pd.Series(equal_returns[bull_days.values]).mean() * 252 * 100,
    'ML Bear': pd.Series(ml_returns[bear_days.values]).mean() * 252 * 100,
    'Equal Bear': pd.Series(equal_returns[bear_days.values]).mean() * 252 * 100
}
pd.Series(data).plot(kind='bar', ax=ax2,
                     color=['blue', 'orange', 'darkblue', 'darkorange'])
ax2.set_title('Annualized Returns by Market Regime')
ax2.set_ylabel('Annual Return %')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)

st.markdown("---")

# Turnover analysis
st.subheader("Transaction Cost & Turnover Analysis")
turnover = ml_weights_weekly.diff().abs().sum(axis=1)
col1, col2, col3 = st.columns(3)
col1.metric("Avg Daily Turnover", f"{turnover.mean():.3f}")
col2.metric("Annual Turnover", f"{turnover.mean() * 252:.1f}x")
col3.metric("Annual Cost Drag", f"{daily_costs.mean() * 252 * 100:.2f}%")

fig3, ax3 = plt.subplots(figsize=(12, 3))
daily_costs.cumsum().plot(ax=ax3, color='red', linewidth=2)
ax3.set_title("Cumulative Transaction Costs")
ax3.set_xlabel("Days")
ax3.grid(True)
st.pyplot(fig3)

st.markdown("---")

# Allocation
st.subheader("Asset Allocation Over Time")
fig4, ax4 = plt.subplots(figsize=(12, 4))
ml_weights_weekly.plot(ax=ax4, kind='area', stacked=True)
ax4.set_xlabel("Days")
ax4.set_ylabel("Weight")
ax4.legend(loc='upper right')
st.pyplot(fig4)

st.markdown("---")

# Key Findings
st.subheader("Key Findings")
st.markdown("""
- 📈 **Bull market alpha** — ML Sharpe 2.51 vs Equal Weight 3.37 in rising markets
- 📉 **Bear market weakness** — ML Sharpe -1.22 vs Equal Weight -1.08 in falling markets  
- 💸 **Transaction costs matter** — 58x annual turnover creates 5.81% cost drag
- ⚖️ **Risk-adjusted** — After costs, equal weight is more efficient overall
""")

st.markdown("---")

# Theory
st.subheader("Theoretical Context")
st.info("""
**Efficient Market Hypothesis (EMH)**

Results suggest markets are difficult to outperform consistently after transaction costs, 
consistent with semi-strong market efficiency. The model captures short-term momentum 
signals but cannot systematically exploit them once realistic trading costs are applied.

This aligns with Fama's EMH — prices already reflect publicly available information, 
making sustained alpha generation extremely difficult.
""")

st.markdown("---")

# Improvements
st.subheader("Iteration & Improvements")
st.success("""
**What we tried and what worked:**

| Improvement | Impact |
|---|---|
| Fixed data leakage (shift features) | More honest, reliable results |
| Risk-adjusted weights (return/volatility) | Sharpe improved 1.88 → 1.99 |
| Weekly rebalancing instead of daily | Cost drag reduced 22% → 5.81% |
| XGBoost hyperparameter tuning | RMSE improved across all assets |

Reducing trading frequency had the single biggest impact on risk-adjusted performance.
""")

st.markdown("---")

# Limitations
st.subheader("Limitations & Future Work")
st.warning("""
**Known Limitations:**

1. **Technical signals only** — Model relies purely on price/volume features. 
   No macro or fundamental signals (earnings, interest rates, sentiment) included.

2. **Bull market bias** — Tested on 2022-2024 which includes a significant tech bull run. 
   Performance in a prolonged bear market is unknown.

3. **Small universe** — Only 6 assets. A larger, more diverse universe may improve 
   diversification and reduce correlation risk.

4. **Perfect execution assumed** — Real trading involves slippage, bid-ask spreads, 
   and liquidity constraints beyond simple transaction costs.

**Future improvements:**
- Add sentiment analysis from financial news
- Include macro features (VIX, interest rates, dollar index)
- Test on 2008 financial crisis data
- Implement proper walk-forward validation
""")