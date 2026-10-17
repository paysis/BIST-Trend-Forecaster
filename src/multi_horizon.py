"""Çoklu vade trend tahmin motoru (1 Gün, 1 Hafta, 1 Ay) ve trend konfluans analizi.

Farklı yatırım vadeleri (günlük swing, haftalık dalga ve aylık ana trend) için
yön tahminleri üretir ve vadelerin birbirini doğrulama derecesini (confluence) puanlar.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src import signals


@dataclass
class HorizonForecast:
    """Tek bir vade için tahmin çıktısını temsil eder."""

    name: str  # "1 Gün", "1 Hafta", "1 Ay"
    days: int  # 1, 5, 20
    probability: float  # 0.0 - 1.0 arası yükseliş olasılığı
    signal: int  # 1 (Al/Yükseliş), 0 (Nötr), -1 (Sat/Düşüş)
    label: str  # "YÜKSELİŞ", "NÖTR", "DÜŞÜŞ"


@dataclass
class MultiHorizonResult:
    """Tüm vadeler için toplu tahmin ve trend uyumu (confluence) özeti."""

    forecasts: Dict[str, HorizonForecast]
    confluence_score: float  # -1.0 (Tam Ayı) ile +1.0 (Tam Boğa) arası
    confluence_label: str  # "GÜÇLÜ BOĞA", "DÜZELTME", "TEPKİ", "AYI", "KARARSIZ"
    summary: str  # Detaylı açıklama metni


HORIZON_CONFIGS = {
    "1d": {"name": "1 Gün (Kısa Vade)", "days": 1},
    "5d": {"name": "1 Hafta (Orta Vade)", "days": 5},
    "20d": {"name": "1 Ay (Uzun Vade)", "days": 20},
}


def add_multi_horizon_targets(
    df: pd.DataFrame,
    close_col: str = "close",
) -> pd.DataFrame:
    """Veri çerçevesine 1 günlük, 5 günlük (1 hafta) ve 20 günlük (1 ay) hedef yönleri ekler.

    target = 1 if close[t + days] > close[t] else 0.
    Gelecek vadeler veri sınırını aştığında NaN olarak işaretlenir.
    """
    if close_col not in df.columns:
        raise ValueError(f"'{close_col}' sütunu veri çerçevesinde bulunamadı.")

    df_out = df.copy()
    close_series = df_out[close_col]

    for key, cfg in HORIZON_CONFIGS.items():
        n = cfg["days"]
        future_close = close_series.shift(-n)
        target_series = (future_close > close_series).astype(float)
        # Sonda kalan n satır için shift NaN üretir; float kalmasını sağlayalım
        target_series[future_close.isna()] = float("nan")
        df_out[f"target_{key}"] = target_series

    return df_out


def compute_confluence(forecasts: Dict[str, HorizonForecast]) -> Tuple[float, str, str]:
    """Farklı vadelerin sinyallerini ve olasılıklarını sentezleyerek konfluans puanı üretir.

    Ağırlıklar:
    - 1 Gün: 0.25 (Taktiksel zamanlama)
    - 1 Hafta: 0.45 (Ana swing dalgası)
    - 1 Ay: 0.30 (Makro trend zemini)

    Döndürülenler:
        (confluence_score, confluence_label, description)
    """
    weights = {"1d": 0.25, "5d": 0.45, "20d": 0.30}

    def _to_num(sig: Any) -> int:
        if isinstance(sig, (int, float)):
            return int(sig)
        if sig == signals.UP:
            return 1
        if sig == signals.DOWN:
            return -1
        return 0

    weighted_score = 0.0
    for key, weight in weights.items():
        if key in forecasts:
            weighted_score += _to_num(forecasts[key].signal) * weight

    sig_1d = _to_num(forecasts.get("1d", HorizonForecast("1d", 1, 0.5, 0, "NÖTR")).signal)
    sig_5d = _to_num(forecasts.get("5d", HorizonForecast("5d", 5, 0.5, 0, "NÖTR")).signal)
    sig_20d = _to_num(forecasts.get("20d", HorizonForecast("20d", 20, 0.5, 0, "NÖTR")).signal)

    # Yorumlama Mantığı
    if sig_1d == 1 and sig_5d == 1 and sig_20d == 1:
        label = "GÜÇLÜ BOĞA (TREND ONAYI)"
        summary = (
            "Kısa, orta ve uzun vadeli tüm modeller eş zamanlı yükseliş teyidi veriyor. "
            "Trend uyumu maksimum seviyede."
        )
    elif sig_1d == -1 and sig_5d == -1 and sig_20d == -1:
        label = "GÜÇLÜ AYI (DÜŞÜŞ ONAYI)"
        summary = (
            "Tüm vadelerde düşüş trendi baskın. Yeni pozisyon açılması yüksek risk "
            "taşımaktadır; nakit korunmalı."
        )
    elif sig_1d == -1 and sig_5d == 1 and sig_20d == 1:
        label = "BOĞA İÇİ SAĞLIKLI DÜZELTME"
        summary = (
            "Orta ve uzun vadeli ana trend yukarı yönlü; 1 günlük düşüş beklentisi "
            "trend içi kâr satışı / düzeltme fırsatı niteliğinde."
        )
    elif sig_1d == 1 and sig_5d == -1 and sig_20d == -1:
        label = "AYI İÇİ TEPKİ YÜKSELİŞİ"
        summary = (
            "Ana trend zayıf ve düşüş yönlü; günlük yükseliş beklentisi geçici tepki "
            "alımı niteliğinde olabilir, stop-loss titizlikle izlenmelidir."
        )
    elif weighted_score > 0.20:
        label = "ILIMLI BOĞA EĞİLİMİ"
        summary = (
            "Modeller genel olarak yukarı yönlü meyilli, ancak vadeler arası kısmi ayrışma var."
        )
    elif weighted_score < -0.20:
        label = "ILIMLI AYI EĞİLİMİ"
        summary = "Modeller genel olarak aşağı yönlü baskı altında, savunmacı strateji önerilir."
    else:
        label = "KARARSIZ / NÖTR TREND"
        summary = "Vadeler birbirini nötrlemekte; belirgin bir yön teyidi bulunmuyor."

    return round(weighted_score, 3), label, summary


def predict_horizons(
    models: Dict[str, Any],
    X: pd.DataFrame,
    prob_low: float = 0.47,
    prob_high: float = 0.53,
    fallback_prob: Optional[float] = None,
) -> MultiHorizonResult:
    """Farklı vadelerdeki modelleri kullanarak çoklu vade tahmin paketini oluşturur.

    Args:
        models: {"1d": model_1d, "5d": model_5d, "20d": model_20d}
        X: Tek satırlık öznitelik DataFrame'i (veya ilk satırı kullanılır)
        prob_low: Nötr / Düşüş eşiği
        prob_high: Nötr / Yükseliş eşiği
        fallback_prob: Model bulunmadığında kullanılacak varsayılan olasılık
    """
    if X.empty:
        raise ValueError("Tahmin yapılacak öznitelik verisi (X) boş olamaz.")

    X_row = X.iloc[[0]]
    forecasts: Dict[str, HorizonForecast] = {}

    for key, cfg in HORIZON_CONFIGS.items():
        model = models.get(key)
        if model is not None:
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_row)[0][1])
            elif callable(model):
                prob = float(model(X_row))
            else:
                prob = 0.50
        elif fallback_prob is not None:
            prob = fallback_prob
        else:
            # Model yoksa nötr kabul edilir
            prob = 0.50

        sig_str = signals.classify_signal(prob, prob_low, prob_high)
        sig_val = 1 if sig_str == signals.UP else (-1 if sig_str == signals.DOWN else 0)
        label = {signals.UP: "YÜKSELİŞ", signals.NEUTRAL: "NÖTR", signals.DOWN: "DÜŞÜŞ"}[sig_str]

        forecasts[key] = HorizonForecast(
            name=cfg["name"],
            days=cfg["days"],
            probability=round(prob, 4),
            signal=sig_val,
            label=label,
        )

    confluence_score, confluence_label, summary = compute_confluence(forecasts)

    return MultiHorizonResult(
        forecasts=forecasts,
        confluence_score=confluence_score,
        confluence_label=confluence_label,
        summary=summary,
    )
