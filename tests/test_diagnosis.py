"""tests/test_diagnosis.py - Hatalı sinyal kök neden analizi birim testleri."""

import pytest

from src.diagnosis import DiagnosisCategory, diagnose_prediction_outcome


def test_diagnose_prediction_outcome_success_bullish():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=0.035,
        index_return=0.01,
        entry_price=100.0,
        exit_price=103.5,
    )
    assert result.is_success is True
    assert result.category == DiagnosisCategory.SUCCESS
    assert "🟢" in result.badge
    assert result.metrics["exit_price"] == pytest.approx(103.5)


def test_diagnose_prediction_outcome_success_bearish():
    result = diagnose_prediction_outcome(
        entry_signal=-1,
        stock_return=-0.025,
        index_return=-0.01,
    )
    assert result.is_success is True
    assert result.category == DiagnosisCategory.SUCCESS


def test_diagnose_prediction_outcome_index_crash():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=-0.04,
        index_return=-0.035,  # % -3.5 BIST 100 çöküşü
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.INDEX_CRASH
    assert "Endeks" in result.badge
    assert "sistemik" in result.explanation


def test_diagnose_prediction_outcome_fx_shock():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=-0.02,
        index_return=-0.005,
        fx_return=0.032,  # % +3.2 USD/TRY sıçraması
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.FX_SHOCK
    assert "Kur Şoku" in result.badge
    assert "Döviz" in result.explanation


def test_diagnose_prediction_outcome_bull_trap():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=-0.015,
        volume_ratio=0.65,  # Hacim ortalamanın %65'i
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.BULL_TRAP
    assert "Boğa Tuzağı" in result.badge


def test_diagnose_prediction_outcome_technical_divergence():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=-0.02,
        volume_ratio=1.1,
        technical_divergence=True,
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.TECHNICAL_DIVERGENCE
    assert "Uyuşmazlık" in result.badge


def test_diagnose_prediction_outcome_stock_idiosyncratic():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=-0.065,  # Hisse %-6.5 düştü
        index_return=0.005,  # Endeks pozitif
        volume_ratio=1.2,
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.STOCK_IDIOSYNCRATIC
    assert "Şirkete Özel" in result.badge


def test_diagnose_prediction_outcome_bearish_failed_index_rally():
    result = diagnose_prediction_outcome(
        entry_signal=-1,
        stock_return=0.04,
        index_return=0.035,  # Endeks rallisi
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.INDEX_RALLY
    assert "Rallisi" in result.badge


def test_diagnose_prediction_outcome_normal_noise():
    result = diagnose_prediction_outcome(
        entry_signal=1,
        stock_return=-0.005,
        index_return=-0.002,
        volume_ratio=1.05,
    )
    assert result.is_success is False
    assert result.category == DiagnosisCategory.NORMAL_MARKET_NOISE
    assert "Olağan" in result.badge
