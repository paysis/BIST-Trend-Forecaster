"""Hatalı sinyal kök neden analizi ve post-mortem teşhis modülü (Explainable AI - XAI).

Bir hisse tahmininin neden gerçekleşmediğini (sistematik endeks düşüşü, makro kur şoku,
hacimsiz boğa tuzağı, teknik uyumsuzluk veya şirkete özgü negatif ayrışma) tespit eder.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DiagnosisCategory(str, Enum):
    """Kök neden tanı kategorileri."""

    SUCCESS = "success"
    INDEX_CRASH = "index_crash"
    INDEX_RALLY = "index_rally"
    FX_SHOCK = "fx_shock"
    BULL_TRAP = "bull_trap"
    TECHNICAL_DIVERGENCE = "technical_divergence"
    STOCK_IDIOSYNCRATIC = "stock_idiosyncratic"
    NORMAL_MARKET_NOISE = "normal_market_noise"


@dataclass
class DiagnosisResult:
    """Teşhis çıktısı veri yapısı."""

    is_success: bool
    category: DiagnosisCategory
    badge: str
    primary_cause: str
    explanation: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def diagnose_prediction_outcome(
    entry_signal: int,
    stock_return: float,
    index_return: Optional[float] = None,
    fx_return: Optional[float] = None,
    volume_ratio: Optional[float] = None,
    technical_divergence: bool = False,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
) -> DiagnosisResult:
    """Bir tahmin sonucunu analiz ederek başarıyı veya başarısızlığın kök nedenini belirler.

    Args:
        entry_signal: 1 (Yükseliş/Al), -1 (Düşüş/Sat), 0 (Nötr)
        stock_return: İlgili vadedeki hisse getirisi (örn: -0.04 -> %-4.00)
        index_return: BIST 100 endeksi getirisi (örn: -0.03 -> %-3.00)
        fx_return: USD/TRY döviz kuru değişimi (örn: +0.035 -> %+3.50)
        volume_ratio: Kırılım günü işlem hacminin 20 günlük ortalamaya oranı
        technical_divergence: RSI/MACD negatif uyumsuzluğu var mı?
        entry_price: Sinyal anındaki fiyat
        exit_price: Vade sonundaki fiyat
    """
    metrics = {
        "stock_return": round(stock_return, 4),
        "index_return": round(index_return, 4) if index_return is not None else None,
        "fx_return": round(fx_return, 4) if fx_return is not None else None,
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "technical_divergence": technical_divergence,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }

    # 1. Başarı Kontrolü
    if (entry_signal == 1 and stock_return > 0) or (entry_signal == -1 and stock_return < 0):
        return DiagnosisResult(
            is_success=True,
            category=DiagnosisCategory.SUCCESS,
            badge="🟢 Başarılı Tahmin",
            primary_cause=f"Hisse öngörülen yönde hareket etti (%{stock_return * 100:+.2f}).",
            explanation="Yapay zeka sinyali piyasa dinamikleriyle örtüştü ve getiri hedefe ulaştı.",
            metrics=metrics,
        )

    # 2. Yükseliş Sinyali Başarısızlığı (Sinyal = 1 ama Hisse Düştü / Getiri <= 0)
    if entry_signal == 1:
        # A) Sistematik Endeks Çöküşü (BIST 100 <= -2.5%)
        if index_return is not None and index_return <= -0.025:
            return DiagnosisResult(
                is_success=False,
                category=DiagnosisCategory.INDEX_CRASH,
                badge="🏛️ Sistematik Endeks Satışı",
                primary_cause=f"BIST 100 endeks çöküşü (%{index_return * 100:.2f})",
                explanation=(
                    f"Hisse teknik olarak yükseliş potansiyeli taşırken BIST 100 genelinde "
                    f"yaşanan %{index_return * 100:.2f}'lik satış dalgası hisseyi aşağı çekti. "
                    "Bu durum modele değil, sistemik piyasa şokuna bağlıdır."
                ),
                metrics=metrics,
            )

        # B) Makro Kur Şoku (USD/TRY >= +2.5%)
        if fx_return is not None and fx_return >= 0.025:
            return DiagnosisResult(
                is_success=False,
                category=DiagnosisCategory.FX_SHOCK,
                badge="💵 Makro Kur Şoku",
                primary_cause=f"USD/TRY ani sıçraması (%{fx_return * 100:+.2f})",
                explanation=(
                    f"Döviz kurunda meydana gelen ani sıçrama (%{fx_return * 100:+.2f}) "
                    "hisse senedi piyasasında faiz ve likidite baskısı oluşturarak "
                    "tahminin gerçekleşmesini engelledi."
                ),
                metrics=metrics,
            )

        # C) Boğa Tuzağı / Düşük Hacimli Sahte Kırılım (Hacim < %80 ortalama)
        if volume_ratio is not None and volume_ratio < 0.80:
            return DiagnosisResult(
                is_success=False,
                category=DiagnosisCategory.BULL_TRAP,
                badge="🪤 Boğa Tuzağı (Hacimsiz Kırılım)",
                primary_cause=f"İşlem hacmi yetersiz (Ortalamanın %{volume_ratio * 100:.0f}'i)",
                explanation=(
                    f"Fiyat yükselişi hacimle desteklenmedi (hacim ortalamanın "
                    f"%{volume_ratio * 100:.0f}'i). Kurumsal alıcı desteği olmayan sahte kırılım "
                    "sonrası satış baskısı geldi."
                ),
                metrics=metrics,
            )

        # D) Teknik Gösterge Uyuşmazlığı (RSI / MACD Sapması)
        if technical_divergence:
            return DiagnosisResult(
                is_success=False,
                category=DiagnosisCategory.TECHNICAL_DIVERGENCE,
                badge="📉 Negatif Teknik Uyuşmazlık",
                primary_cause="RSI/Momentum negatif uyumsuzluğu",
                explanation=(
                    "Fiyat tepeleri gösterge tepeleriyle desteklenmedi (negatif divergence). "
                    "Momentum zayıflaması alıcıların tükenmesine yol açtı."
                ),
                metrics=metrics,
            )

        # E) Şirkete Özel Negatif Ayrışma (Endeks yatay/pozitifken hisse <= -5%)
        if stock_return <= -0.05 and (index_return is None or index_return > -0.015):
            return DiagnosisResult(
                is_success=False,
                category=DiagnosisCategory.STOCK_IDIOSYNCRATIC,
                badge="🏢 Şirkete Özel Negatif Ayrışma",
                primary_cause=f"Hisse endeksten bağımsız sert düştü (%{stock_return * 100:.2f})",
                explanation=(
                    "Endeks yatay veya pozitif seyrederken hissenin sert değer kaybetmesi, "
                    "şirkete özgü (KAP haberi, finansal tablo, ortak satışı vb.) dışsal bir olaya "
                    "işaret eder."
                ),
                metrics=metrics,
            )

    # 3. Düşüş Sinyali Başarısızlığı (Sinyal = -1 ama Hisse Yükseldi)
    if entry_signal == -1:
        if index_return is not None and index_return >= 0.025:
            return DiagnosisResult(
                is_success=False,
                category=DiagnosisCategory.INDEX_RALLY,
                badge="🏛️ Genel Endeks Rallisi",
                primary_cause=f"BIST 100 güçlü yükselişi (%{index_return * 100:+.2f})",
                explanation=(
                    f"Genel endekste başlayan sert ralli (%{index_return * 100:+.2f}) "
                    "zayıf trenddeki hisseleri de beraberinde yukarı taşıdı."
                ),
                metrics=metrics,
            )

    # 4. Varsayılan Piyasa Gürültüsü
    return DiagnosisResult(
        is_success=False,
        category=DiagnosisCategory.NORMAL_MARKET_NOISE,
        badge="⏸️ Olağan Piyasa Dalgalanması",
        primary_cause=f"Kısa vadeli olağan dalgalanma (%{stock_return * 100:+.2f})",
        explanation=(
            "Belirgin bir makro veya sistemik şok saptanmadı; tahmin piyasa gürültüsü "
            "ve normal destek/direnç salınımları sınırında kaldı."
        ),
        metrics=metrics,
    )
