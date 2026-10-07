# tests/test_backtest.py - backtest.py birim testleri
import numpy as np
import pandas as pd
import pytest

from src import backtest

DATES = pd.bdate_range("2026-01-05", periods=4)


@pytest.fixture
def close():
    # Günlük getiriler: +%10, -%10, %0
    return pd.Series([100.0, 110.0, 99.0, 99.0], index=DATES)


def test_strategy_earns_next_day_return_of_todays_position(close):
    positions = pd.Series([1, 0, 1, 1], index=DATES)
    result = backtest.strategy_returns(close, positions)

    # Getiriler gerçekleştikleri güne yazılır; ilk günün getirisi yoktur
    assert list(result.index) == list(DATES[1:])
    assert result["buy_hold"].tolist() == pytest.approx([0.10, -0.10, 0.0])
    assert result["strategy"].tolist() == pytest.approx([0.10, 0.0, 0.0])


def test_last_position_is_not_used_without_next_close(close):
    """Bugünün sinyalinin getirisi henüz bilinmez; sonuca katılmamalı (look-ahead yok)."""
    a = backtest.strategy_returns(close, pd.Series([1, 0, 1, 0], index=DATES))
    b = backtest.strategy_returns(close, pd.Series([1, 0, 1, 1], index=DATES))
    pd.testing.assert_frame_equal(a, b)


def test_past_returns_do_not_depend_on_future_prices(close):
    later = close.copy()
    later.iloc[-1] = 500.0
    positions = pd.Series([1, 1, 1, 1], index=DATES)
    a = backtest.strategy_returns(close, positions)
    b = backtest.strategy_returns(later, positions)
    pd.testing.assert_frame_equal(a.iloc[:-1], b.iloc[:-1])


def test_trading_cost_is_charged_on_each_position_change(close):
    # Nakitte başlar: 0->1 (t0), 1->0 (t1), 0->1 (t2) = 3 işlem
    positions = pd.Series([1, 0, 1, 1], index=DATES)
    result = backtest.strategy_returns(close, positions, cost=0.01)
    assert result["strategy"].tolist() == pytest.approx([0.09, -0.01, -0.01])


def test_cumulative_return_compounds():
    returns = pd.Series([0.10, -0.10, 0.0])
    assert backtest.cumulative_returns(returns).tolist() == pytest.approx([0.10, -0.01, -0.01])


def test_max_drawdown_from_running_peak():
    assert backtest.max_drawdown(pd.Series([0.10, -0.10, 0.0])) == pytest.approx(-0.10)


def test_max_drawdown_counts_loss_from_initial_capital():
    assert backtest.max_drawdown(pd.Series([-0.20, 0.10])) == pytest.approx(-0.20)


def test_max_drawdown_is_zero_when_equity_never_falls():
    assert backtest.max_drawdown(pd.Series([0.01, 0.0, 0.02])) == 0.0


def test_sharpe_ratio_is_annualised_mean_over_std():
    returns = pd.Series([0.01, -0.01, 0.02])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert backtest.sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_ratio_undefined_without_volatility():
    assert np.isnan(backtest.sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))


def test_summarize_reports_total_return_sharpe_drawdown():
    returns = pd.Series([0.10, -0.10, 0.0])
    summary = backtest.summarize(returns)
    assert summary["total_return"] == pytest.approx(-0.01)
    assert summary["max_drawdown"] == pytest.approx(-0.10)
    assert summary["sharpe"] == pytest.approx(backtest.sharpe_ratio(returns))


def test_run_backtest_holds_only_on_up_signals(close):
    frame = pd.DataFrame({"Date": DATES, "close": close.values})
    probs = np.array([0.60, 0.50, 0.60, 0.40])

    result = backtest.run_backtest(frame, probs, threshold=0.55)

    assert result.returns["strategy"].tolist() == pytest.approx([0.10, 0.0, 0.0])
    assert result.curve["strategy"].tolist() == pytest.approx([0.0, 0.10, 0.10, 0.10])
    assert result.curve["buy_hold"].tolist() == pytest.approx([0.0, 0.10, -0.01, -0.01])
    assert list(result.curve.index) == list(DATES)
    assert result.strategy["total_return"] == pytest.approx(0.10)
    assert result.buy_hold["total_return"] == pytest.approx(-0.01)
    # Gerçekleşen 3 günün 2'sinde pozisyondaydı
    assert result.exposure == pytest.approx(2 / 3)
    assert result.trades == 3


def test_run_backtest_requires_two_days():
    frame = pd.DataFrame({"Date": DATES[:1], "close": [100.0]})
    with pytest.raises(ValueError, match="en az 2"):
        backtest.run_backtest(frame, np.array([0.6]), threshold=0.55)


def test_recent_window_keeps_last_months():
    dates = pd.bdate_range("2025-01-01", "2025-12-31")
    frame = pd.DataFrame({"Date": dates, "close": 1.0})
    window = backtest.recent_window(frame, months=6)
    assert window["Date"].min() > pd.Timestamp("2025-06-30")
    assert window["Date"].max() == dates[-1]
    assert list(window.index) == list(range(len(window)))
