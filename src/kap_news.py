"""BIST KAP (Kamuyu Aydınlatma Platformu) ve şirket haber akışı analiz modülü.

Hisselere ait son bildirimleri ayrıştırır, olumlu/olumsuz olay kategorizasyonu yapar
ve teknik model tahminlerine temel analiz desteği sağlar.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

from src import config


class EventCategory(str, Enum):
    """KAP bildirim duygu ve olay kategorisi."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class KAPDisclosure:
    """Tek bir KAP bildirimi veya şirket haberini temsil eder."""

    ticker: str
    title: str
    summary: str
    publish_date: str
    category: EventCategory
    badge: str
    event_type: str


# Kural tabanlı Türkçe anahtar kelime eşlemeleri
POSITIVE_KEYWORDS = [
    ("pay geri alım", "🟢 Pay Geri Alımı", "pay_geri_alimi"),
    ("hisse geri alım", "🟢 Pay Geri Alımı", "pay_geri_alimi"),
    ("yeni iş ilişkisi", "🟢 Yeni İş İlişkisi", "yeni_is"),
    ("ihale", "🟢 İhale Kazanımı", "yeni_is"),
    ("sipariş", "🟢 Yeni Sipariş", "yeni_is"),
    ("sözleşme imzalan", "🟢 Sözleşme İmzalanması", "yeni_is"),
    ("temettü", "🟢 Temettü Dağıtımı", "temettu"),
    ("kar payı", "🟢 Temettü Dağıtımı", "temettu"),
    ("kâr artı", "🟢 Kâr Artışı", "kar_artisi"),
    ("rekor kâr", "🟢 Rekor Kâr", "kar_artisi"),
    ("bedelsiz", "🟢 Bedelsiz Sermaye Artırımı", "bedelsiz"),
]

NEGATIVE_KEYWORDS = [
    ("spk", "🔴 SPK / VBTS Tedbiri", "spk_tedbiri"),
    ("tedbir", "🔴 Tedbir Uygulaması", "spk_tedbiri"),
    ("kredili işlem yasağı", "🔴 Kredili İşlem Yasağı", "spk_tedbiri"),
    ("brüt takas", "🔴 Brüt Takas Tedbiri", "spk_tedbiri"),
    ("vbts", "🔴 VBTS Tedbiri", "spk_tedbiri"),
    ("pay satış", "🔴 Pay Satış Bildirimi", "pay_satisi"),
    ("ortak satış", "🔴 Ortak Pay Satışı", "pay_satisi"),
    ("bedelli sermaye", "🔴 Bedelli Sermaye Artırımı", "bedelli"),
    ("sermaye azalt", "🔴 Sermaye Azaltımı", "bedelli"),
    ("zarar", "🔴 Zarar / Kâr Düşüşü", "zarar"),
    ("dava", "🔴 Dava / İhtilaf Bildirimi", "hukuki"),
    ("soruşturma", "🔴 Soruşturma Bildirimi", "hukuki"),
    ("konkordato", "🔴 Konkordato / Finansal Zorluk", "iflas_riski"),
]

NEUTRAL_KEYWORDS = [
    ("genel kurul", "⚪ Olağan Genel Kurul", "genel_kurul"),
    ("faaliyet raporu", "⚪ Finansal Rapor Açıklaması", "rapor"),
    ("bağımsız denetim", "⚪ Denetim Raporu", "rapor"),
    ("kayıtlı sermaye", "⚪ Kayıtlı Sermaye Tavanı", "sermaye_tavan"),
    ("derecelendirme", "⚪ Kredi Derecelendirmesi", "derecelendirme"),
]


def normalize_turkish_text(text: str) -> str:
    """Türkçe büyük/küçük harf duyarlılığını ve aksanları standartlaştırır."""
    rep = {
        "İ": "i",
        "I": "ı",
        "i̇": "i",
        "\u0307": "",
        "â": "a",
        "Â": "a",
        "î": "i",
        "Î": "i",
        "û": "u",
        "Û": "u",
    }
    for k, v in rep.items():
        text = text.replace(k, v)
    return text.lower().replace("\u0307", "")


def classify_disclosure(title: str, summary: str = "") -> Tuple[EventCategory, str, str]:
    """Başlık ve özet metnini analiz ederek duygu kategorisi, rozet ve olay türü döner."""
    text = normalize_turkish_text(f"{title} {summary}")

    # Önce kritik negatifleri kontrol et
    for kw, badge, event_type in NEGATIVE_KEYWORDS:
        if normalize_turkish_text(kw) in text:
            return EventCategory.NEGATIVE, badge, event_type

    # Ardından pozitif olayları kontrol et
    for kw, badge, event_type in POSITIVE_KEYWORDS:
        if normalize_turkish_text(kw) in text:
            return EventCategory.POSITIVE, badge, event_type

    # Kurumsal olağan bildirimler
    for kw, badge, event_type in NEUTRAL_KEYWORDS:
        if normalize_turkish_text(kw) in text:
            return EventCategory.NEUTRAL, badge, event_type

    return EventCategory.NEUTRAL, "⚪ Genel Bilgilendirme", "diger"


def parse_raw_news_item(ticker: str, item: Dict[str, Any]) -> KAPDisclosure:
    """Harici haber API'sinden gelen sözlük yapısını KAPDisclosure nesnesine dönüştürür."""
    # Yahoo Finance haber yapısı (content veya title)
    title = (
        item.get("title")
        or (item.get("content", {}).get("title") if isinstance(item.get("content"), dict) else "")
        or "Bildirim Başlığı Yok"
    )
    summary = (
        item.get("summary")
        or (item.get("content", {}).get("summary") if isinstance(item.get("content"), dict) else "")
        or ""
    )
    pub_date = str(
        item.get("providerPublishTime") or item.get("pubDate") or item.get("date") or "-"
    )

    category, badge, event_type = classify_disclosure(title, summary)

    return KAPDisclosure(
        ticker=ticker.replace(".IS", ""),
        title=title,
        summary=summary,
        publish_date=pub_date,
        category=category,
        badge=badge,
        event_type=event_type,
    )


def summarize_disclosures(disclosures: List[KAPDisclosure]) -> Dict[str, Any]:
    """Bildirim listesini özetleyerek duygu dengesi ve risk uyarısı üretir."""
    if not disclosures:
        return {
            "total": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "overall_sentiment": "NÖTR / BİLDİRİM YOK",
            "has_critical_negative": False,
            "warning_note": "Son günlerde dikkat çeken bir KAP bildirimi bulunmuyor.",
        }

    pos = sum(1 for d in disclosures if d.category == EventCategory.POSITIVE)
    neg = sum(1 for d in disclosures if d.category == EventCategory.NEGATIVE)
    neu = sum(1 for d in disclosures if d.category == EventCategory.NEUTRAL)

    has_critical = any(
        d.event_type in ("spk_tedbiri", "pay_satisi", "bedelli", "zarar") for d in disclosures
    )

    if neg > pos:
        sentiment = "NEGATİF BASKILI"
        warning = (
            "⚠️ Şirket hakkında negatif bildirimler tespit edildi. "
            "Teknik alım sinyallerinde temkinli olunmalı."
        )
    elif pos > neg:
        sentiment = "POZİTİF DESTEKLİ"
        warning = "🟢 Şirket lehine olumlu KAP haber akışı teknik görünümü destekliyor."
    else:
        sentiment = "DENGELİ / NÖTR"
        warning = "Haber akışı dengeli; teknik göstergeler ön planda izlenmeli."

    return {
        "total": len(disclosures),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "overall_sentiment": sentiment,
        "has_critical_negative": has_critical,
        "warning_note": warning,
    }


def fetch_latest_disclosures(
    ticker: str,
    limit: int = 5,
    custom_fetcher: Optional[Any] = None,
) -> List[KAPDisclosure]:
    """Hisse için en güncel haber ve KAP bildirimlerini çeker.

    Hata veya ağ kesintisi anında uygulamayı çökertmeden boş liste döner (graceful fallback).
    """
    if custom_fetcher is not None:
        try:
            raw_items = custom_fetcher(ticker, limit)
            return [parse_raw_news_item(ticker, item) for item in raw_items[:limit]]
        except Exception:
            return []

    mapped = config.TICKER_YAHOO_MAP.get(ticker, ticker)
    try:
        t = yf.Ticker(mapped)
        news = getattr(t, "news", [])
        if not news:
            return []
        return [parse_raw_news_item(ticker, item) for item in news[:limit]]
    except Exception:
        return []
