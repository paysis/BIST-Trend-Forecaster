# tests/test_simulator.py
"""TL Bazlı What-If Yatırım ve Portföy Büyüme Simülatörü birim testleri (#52)."""

import pandas as pd
import pytest

from src import simulator


@pytest.fixture
def sample_price_frame():
    dates = pd.bdate_range("2025-01-01", periods=5).strftime("%Y-%m-%d")
    prices = [100.0, 110.0, 105.0, 115.0, 120.0]
    return pd.DataFrame({"Date": dates, "close": prices})


def test_simulate_investment_always_bullish(sample_price_frame):
    """Tüm günlerde alım sinyali olduğunda model Al-Tut stratejisini izlemelidir."""
    probs = [0.60, 0.70, 0.65, 0.80, 0.75]
    result = simulator.simulate_investment(
        sample_price_frame, probs, threshold=0.55, initial_capital=100_000.0, cost=0.001
    )

    assert result is not None
    assert result.initial_capital == 100_000.0
    assert result.trades_count == 1  # Sadece ilk gün alış
    assert result.total_return_pct > 15.0
    assert result.final_value == pytest.approx(result.buy_hold_final_value, abs=1.0)
    assert result.alpha_tl == pytest.approx(0.0, abs=1.0)
    assert len(result.daily_timeline) == 5


def test_simulate_investment_always_bearish(sample_price_frame):
    """Tüm günlerde düşüş sinyali olduğunda nakitte kalınmalı ve sermaye korunmalıdır."""
    probs = [0.40, 0.35, 0.45, 0.30, 0.40]
    result = simulator.simulate_investment(
        sample_price_frame, probs, threshold=0.53, initial_capital=100_000.0, cost=0.001
    )

    assert result is not None
    assert result.trades_count == 0
    assert result.final_value == 100_000.0
    assert result.net_profit_tl == 0.0
    assert result.total_return_pct == 0.0
    assert result.total_commission_paid == 0.0
    assert result.max_drawdown_pct == 0.0
    # Al-tut hissede kaldığı için değer kazanmış ama model nakitte kalmış
    assert result.buy_hold_final_value > 100_000.0


def test_simulate_investment_avoids_crash(sample_price_frame):
    """Model düşüş günlerinde nakde geçerek Al-Tut'a kıyasla pozitif alpha üretmelidir."""
    # Fiyat serisi: 100 -> 120 (yükseliş) -> 70 (çöküş) -> 90 (toparlanma)
    dates = pd.bdate_range("2025-01-01", periods=4).strftime("%Y-%m-%d")
    df_crash = pd.DataFrame({"Date": dates, "close": [100.0, 120.0, 70.0, 90.0]})

    # 1. gün al, 2. gün sat (120'den çık), 3. gün nakitte kal (70'e düşüşü atlat), 4. gün al
    probs = [0.60, 0.40, 0.40, 0.60]

    result = simulator.simulate_investment(
        df_crash, probs, threshold=0.53, initial_capital=100_000.0, cost=0.001
    )

    assert result is not None
    # Model 120'de sattığı için 70 çöküşünü atlattı
    assert result.final_value > result.buy_hold_final_value
    assert result.alpha_tl > 0.0
    assert result.trades_count == 3  # Al -> Sat -> Al


def test_simulate_investment_commission_effect(sample_price_frame):
    """Yüksek komisyon oranı net getiriyi düşürmeli ve ödenen komisyonu artırmalıdır."""
    probs = [0.60, 0.40, 0.60, 0.40, 0.60]

    res_low_cost = simulator.simulate_investment(
        sample_price_frame, probs, threshold=0.53, initial_capital=100_000.0, cost=0.0005
    )
    res_high_cost = simulator.simulate_investment(
        sample_price_frame, probs, threshold=0.53, initial_capital=100_000.0, cost=0.005
    )

    assert res_high_cost.total_commission_paid > res_low_cost.total_commission_paid
    assert res_high_cost.final_value < res_low_cost.final_value


def test_simulate_investment_validation_errors(sample_price_frame):
    """Geçersiz girdilerde ValueError veya None döndürülmelidir."""
    probs = [0.60, 0.70, 0.65, 0.80, 0.75]

    # Sıfır veya negatif sermaye
    with pytest.raises(ValueError, match="sıfırdan büyük"):
        simulator.simulate_investment(sample_price_frame, probs, initial_capital=0.0)

    with pytest.raises(ValueError, match="sıfırdan büyük"):
        simulator.simulate_investment(sample_price_frame, probs, initial_capital=-50_000.0)

    # Olasılık uzunluğu uyuşmazlığı
    with pytest.raises(ValueError, match="uzunluğu"):
        simulator.simulate_investment(sample_price_frame, [0.60, 0.70])

    # Boş DataFrame
    assert simulator.simulate_investment(pd.DataFrame(), []) is None
