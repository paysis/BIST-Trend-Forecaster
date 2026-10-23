# src/risk_guard.py
"""Piyasa Şoku ve Aşırı Oynaklık Koruma Kalkanı (Risk Guard / Circuit Breaker) (#54).

Borsa İstanbul'da olağanüstü satış dalgaları, makro kur sıçramaları veya devre kesici
şokları patlak verdiğinde modelin alım sinyallerini otomatik olarak savunma / nakit
moduna geçiren koruma katmanıdır.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ShockAssessment:
    """Piyasa şoku ve olağanüstü oynaklık değerlendirme sonucu."""

    is_shock: bool
    severity: str  # "Normal", "Uyarı", "Kritik Şok"
    reasons: List[str] = field(default_factory=list)
    primary_reason: str = "Normal Piyasa Koşulları"
    xu100_daily_return: float = 0.0
    usdtry_daily_change: float = 0.0
    ticker_daily_return: Optional[float] = None


def detect_market_shock(
    df_macro: Optional[pd.DataFrame] = None,
    ticker_df: Optional[pd.DataFrame] = None,
    xu_drop_threshold: float = -0.030,  # BIST 100 günlük -%3.0 çöküş
    xu_cumulative_drop_threshold: float = -0.050,  # BIST 100 2 günlük -%5.0 çöküş
    fx_jump_threshold: float = 0.025,  # USD/TRY tek günde +%2.5 sıçrama
    ticker_drop_threshold: float = -0.070,  # Hissede taban riski (-%7.0)
) -> ShockAssessment:
    """Makro göstergeler ve hisse verisini inceleyerek piyasa şoku durumunu tespit eder."""
    reasons = []
    is_shock = False
    severity = "Normal"

    xu_ret = 0.0
    fx_change = 0.0
    ticker_ret: Optional[float] = None

    # 1. Makro Endeks ve Döviz Şoku Analizi
    if df_macro is not None and not df_macro.empty:
        # Son günün getirileri
        last_row = df_macro.iloc[-1]
        xu_ret = float(last_row.get("xu100_ret", 0.0))
        fx_change = float(last_row.get("usdtry_change", 0.0))

        # Tek günlük sert BIST 100 endeks düşüşü
        if xu_ret <= xu_drop_threshold:
            reasons.append(f"BIST 100 Endeks Şoku (Günlük %{xu_ret * 100:+.2f})")
            is_shock = True

        # İki günlük kümülatif BIST 100 endeks çöküşü
        if len(df_macro) >= 2 and "xu100_ret" in df_macro.columns:
            cum_xu = float(df_macro["xu100_ret"].iloc[-2:].sum())
            if cum_xu <= xu_cumulative_drop_threshold:
                reasons.append(f"BIST 100 2 Günlük Kümülatif Çöküş (%{cum_xu * 100:+.2f})")
                is_shock = True

        # USD/TRY döviz kuru ani sıçraması
        if fx_change >= fx_jump_threshold:
            reasons.append(f"Döviz Kuru Sıçraması (USD/TRY %{fx_change * 100:+.2f})")
            is_shock = True

    # 2. İlgili Hissede Aşırı Satış / Taban Riski Analizi
    if ticker_df is not None and not ticker_df.empty and "close" in ticker_df.columns:
        if len(ticker_df) >= 2:
            closes = ticker_df["close"].to_numpy(dtype=float)
            ticker_ret = float((closes[-1] - closes[-2]) / closes[-2])
            if ticker_ret <= ticker_drop_threshold:
                reasons.append(f"Hissede Aşırı Satış Baskısı (%{ticker_ret * 100:+.2f})")
                is_shock = True

    if is_shock:
        severity = "Kritik Şok" if len(reasons) >= 2 or xu_ret <= -0.050 else "Uyarı"
        primary = reasons[0]
    else:
        primary = "Normal Piyasa Koşulları"

    return ShockAssessment(
        is_shock=is_shock,
        severity=severity,
        reasons=reasons,
        primary_reason=primary,
        xu100_daily_return=round(xu_ret, 4),
        usdtry_daily_change=round(fx_change, 4),
        ticker_daily_return=round(ticker_ret, 4) if ticker_ret is not None else None,
    )


def apply_risk_guard(
    signal: int,
    prob: float,
    assessment: ShockAssessment,
) -> Tuple[int, float, str]:
    """Piyasa şoku algılandığında alım sinyalini ezerek nakit korumasına çeker.

    Döndürülen değerler:
        (efektif_sinyal, efektif_olasılık, durum_açıklaması)
    """
    if assessment.is_shock:
        if signal == 1:
            # Yükseliş sinyali şok nedeniyle geçersiz kılınır
            note = (
                f"⚠️ PİYASA ŞOKU KORUMASI DEVREDE: {assessment.primary_reason}. "
                "Yapay zeka alım sinyali sermaye koruma amacıyla geçici olarak "
                "NAKİT moduna çekildi."
            )
            return 0, min(float(prob), 0.50), note
        else:
            note = f"⚠️ PİYASA ŞOKU: {assessment.primary_reason}. Nakit / savunma pozisyonu korundu."
            return 0, float(prob), note

    return signal, float(prob), "Piyasa koşulları normal seyrediyor."
