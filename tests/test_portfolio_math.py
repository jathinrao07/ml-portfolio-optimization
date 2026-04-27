import numpy as np
import pandas as pd


def compute_weights_like_app(pred_returns: pd.DataFrame, volatility_10: pd.DataFrame, max_weight: float):
    assets = list(pred_returns.columns)
    weights = pd.DataFrame(index=pred_returns.index, columns=assets, dtype=float)

    for i in range(len(pred_returns)):
        row = pred_returns.iloc[i]
        vol = volatility_10.iloc[i].replace(0, np.nan)
        vol = vol.fillna(vol.mean())

        signal = row / vol
        positive = signal[signal > 0]

        if len(positive) == 0:
            w = pd.Series(1 / len(assets), index=assets)
        else:
            raw = (positive / positive.sum()).copy()
            capped = pd.Series(0.0, index=raw.index, dtype=float)
            remaining = raw.copy()

            while True:
                over = remaining[remaining > max_weight]
                if over.empty:
                    break

                capped.loc[over.index] = max_weight
                remaining = remaining.drop(index=over.index)

                leftover = 1.0 - capped.sum()
                if leftover <= 0 or remaining.empty:
                    remaining[:] = 0.0
                    break

                remaining = remaining / remaining.sum() * leftover

            capped.loc[remaining.index] = remaining

            full_w = pd.Series(0.0, index=assets)
            full_w[capped.index] = capped
            w = full_w / full_w.sum()

        weights.iloc[i] = w

    return weights


def test_weights_sum_to_one_and_respect_max_weight():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    pred = pd.DataFrame(
        {
            "AAPL": [0.02, 0.01, -0.01, 0.03, 0.02],
            "MSFT": [0.01, 0.00, -0.02, 0.01, 0.01],
            "SPY": [0.00, 0.02, -0.01, 0.01, 0.00],
        },
        index=idx,
    )
    vol = pd.DataFrame(
        {
            "AAPL": [0.02, 0.02, 0.02, 0.02, 0.02],
            "MSFT": [0.01, 0.01, 0.01, 0.01, 0.01],
            "SPY": [0.015, 0.015, 0.015, 0.015, 0.015],
        },
        index=idx,
    )

    w = compute_weights_like_app(pred, vol, max_weight=0.6)

    row_sums = w.sum(axis=1).values
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-9)
    assert (w.max(axis=1) <= 0.6 + 1e-12).all()


def test_all_negative_signals_falls_back_to_equal_weight():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    pred = pd.DataFrame(
        {
            "AAPL": [-0.01, -0.02],
            "MSFT": [-0.03, -0.01],
            "SPY": [-0.01, -0.01],
        },
        index=idx,
    )
    vol = pd.DataFrame(0.02, index=idx, columns=pred.columns)

    w = compute_weights_like_app(pred, vol, max_weight=0.6)
    assert np.allclose(w.values, np.ones_like(w.values) / w.shape[1])
