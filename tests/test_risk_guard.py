# tests/test_risk_guard.py
"""Piyasa Şoku ve Aşırı Oynaklık Koruma Kalkanı (Risk Guard) birim testleri (#54)."""

import pandas as pd

from src import risk_guard


def test_detect_market_shock_normal_conditions():
    """Piyasa koşulları normal sınırlardayken şok algılanmamalıdır."""
    df_macro = pd.DataFrame(
        [
            {"xu100_ret": 0.008, "usdtry_change": -0.002},
            {"xu100_ret": -0.005, "usdtry_change": 0.004},
        ]
    )
    df_ticker = pd.DataFrame({"close": [100.0, 101.5]})

    assessment = risk_guard.detect_market_shock(df_macro=df_macro, ticker_df=df_ticker)
    assert not assessment.is_shock
    assert assessment.severity == "Normal"
    assert "Normal" in assessment.primary_reason


def test_detect_market_shock_index_single_day_crash():
    """BIST 100 endeksinde tek günde -%3'ten fazla düşüş şok olarak algılanmalıdır."""
    df_macro = pd.DataFrame([{"xu100_ret": -0.038, "usdtry_change": 0.005}])

    assessment = risk_guard.detect_market_shock(df_macro=df_macro)
    assert assessment.is_shock
    assert "Endeks Şoku" in assessment.primary_reason
    assert assessment.xu100_daily_return == -0.038


def test_detect_market_shock_index_cumulative_crash():
    """2 günlük kümülatif BIST 100 düşüşü -%5'i aştığında şok tetiklenmelidir."""
    df_macro = pd.DataFrame(
        [
            {"xu100_ret": -0.027, "usdtry_change": 0.002},
            {"xu100_ret": -0.026, "usdtry_change": 0.003},
        ]
    )

    assessment = risk_guard.detect_market_shock(df_macro=df_macro)
    assert assessment.is_shock
    assert any("2 Günlük Kümülatif" in r for r in assessment.reasons)


def test_detect_market_shock_fx_jump():
    """USD/TRY kurunda tek günde +%2.5'ten fazla sıçrama şok tetiklemelidir."""
    df_macro = pd.DataFrame([{"xu100_ret": -0.005, "usdtry_change": 0.034}])

    assessment = risk_guard.detect_market_shock(df_macro=df_macro)
    assert assessment.is_shock
    assert "Döviz Kuru Sıçraması" in assessment.primary_reason


def test_detect_market_shock_ticker_limit_down():
    """Hissede tek günde -%7'den fazla sert satış şok tetiklemelidir."""
    df_ticker = pd.DataFrame({"close": [100.0, 92.0]})  # -%8.0 düşüş

    assessment = risk_guard.detect_market_shock(ticker_df=df_ticker)
    assert assessment.is_shock
    assert "Hissede Aşırı Satış Baskısı" in assessment.primary_reason
    assert assessment.ticker_daily_return == -0.08


def test_apply_risk_guard_overrides_bullish_signal():
    """Şok durumunda yükseliş sinyali (1) otomatik olarak nakde (0) çekilmelidir."""
    shock = risk_guard.ShockAssessment(
        is_shock=True,
        severity="Kritik Şok",
        primary_reason="BIST 100 Endeks Şoku",
        reasons=["BIST 100 Endeks Şoku"],
    )

    eff_sig, eff_prob, note = risk_guard.apply_risk_guard(signal=1, prob=0.72, assessment=shock)
    assert eff_sig == 0
    assert eff_prob <= 0.50
    assert "PİYASA ŞOKU KORUMASI DEVREDE" in note


def test_apply_risk_guard_leaves_normal_signals_intact():
    """Şok yokken orijinal sinyal ve olasılık aynen korunmalıdır."""
    normal = risk_guard.ShockAssessment(is_shock=False, severity="Normal")

    eff_sig, eff_prob, note = risk_guard.apply_risk_guard(signal=1, prob=0.65, assessment=normal)
    assert eff_sig == 1
    assert eff_prob == 0.65
    assert "normal" in note.lower()


def test_detect_market_shock_handles_empty_inputs():
    """Boş girişler güvenle şok yok (Normal) olarak değerlendirilmelidir."""
    assessment = risk_guard.detect_market_shock(df_macro=None, ticker_df=None)
    assert not assessment.is_shock
    assert assessment.severity == "Normal"

    empty_assessment = risk_guard.detect_market_shock(
        df_macro=pd.DataFrame(), ticker_df=pd.DataFrame()
    )
    assert not empty_assessment.is_shock
