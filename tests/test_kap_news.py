"""tests/test_kap_news.py - BIST KAP bildirimleri ve haber akışı modülü birim testleri."""

from src.kap_news import (
    EventCategory,
    classify_disclosure,
    fetch_latest_disclosures,
    parse_raw_news_item,
    summarize_disclosures,
)


def test_classify_disclosure_positive_events():
    pos_samples = [
        (
            "Şirketimiz Yönetim Kurulu Pay Geri Alım Programı Başlatılmasına Karar Verdi",
            "pay_geri_alimi",
        ),
        ("Yeni İş İlişkisi: 50 Milyon Dolarlık İhale Sözleşmesi İmzalandı", "yeni_is"),
        ("Kâr Payı Dağıtımına (Temettü) İlişkin Yönetim Kurulu Kararı", "temettu"),
        ("Şirketimizin 2024 Yılı Kâr Artışı %85 Olarak Gerçekleşti", "kar_artisi"),
        ("Sermaye Piyasası Kurulu'na Bedelsiz Sermaye Artırımı Başvurusu", "bedelsiz"),
    ]
    for title, expected_type in pos_samples:
        cat, badge, event_type = classify_disclosure(title)
        assert cat == EventCategory.POSITIVE
        assert "🟢" in badge
        assert event_type == expected_type


def test_classify_disclosure_negative_events():
    neg_samples = [
        ("Borsa İstanbul ve SPK Tarafından VBTS Kapsamında Tedbir Uygulanması", "spk_tedbiri"),
        ("Hakim Ortak Tarafından Pay Satış Bildirimi", "pay_satisi"),
        ("Şirketimizin %100 Bedelli Sermaye Artırımı Kararı", "bedelli"),
        ("2024 Yılı 4. Çeyrek Net Dönem Zararı Açıklandı", "zarar"),
    ]
    for title, expected_type in neg_samples:
        cat, badge, event_type = classify_disclosure(title)
        assert cat == EventCategory.NEGATIVE
        assert "🔴" in badge
        assert event_type == expected_type


def test_classify_disclosure_neutral_events():
    cat, badge, event_type = classify_disclosure("2024 Yılı Olağan Genel Kurul Toplantısı Çağrısı")
    assert cat == EventCategory.NEUTRAL
    assert "⚪" in badge
    assert event_type == "genel_kurul"


def test_parse_raw_news_item():
    raw_item = {
        "title": "THYAO Yeni Uçak Siparişi İçin Sözleşme İmzaladı",
        "summary": "Türk Hava Yolları 10 yeni geniş gövde uçak siparişi verdi.",
        "providerPublishTime": 1740000000,
    }
    disc = parse_raw_news_item("THYAO.IS", raw_item)
    assert disc.ticker == "THYAO"
    assert disc.category == EventCategory.POSITIVE
    assert disc.event_type == "yeni_is"
    assert "Yeni" in disc.badge


def test_summarize_disclosures():
    sample_discs = [
        parse_raw_news_item("GARAN", {"title": "Yeni İş İlişkisi Sözleşmesi"}),
        parse_raw_news_item("GARAN", {"title": "Pay Geri Alım İşlemleri"}),
        parse_raw_news_item("GARAN", {"title": "SPK Tedbiri Uygulandı"}),
    ]
    summary = summarize_disclosures(sample_discs)
    assert summary["total"] == 3
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert summary["overall_sentiment"] == "POZİTİF DESTEKLİ"
    assert summary["has_critical_negative"] is True


def test_summarize_disclosures_empty():
    summary = summarize_disclosures([])
    assert summary["total"] == 0
    assert summary["overall_sentiment"] == "NÖTR / BİLDİRİM YOK"
    assert summary["has_critical_negative"] is False


def test_fetch_latest_disclosures_with_custom_fetcher():
    def mock_fetcher(ticker, limit):
        return [
            {"title": f"{ticker} Pay Geri Alımı Bildirdi"},
            {"title": f"{ticker} Olağan Genel Kurul Toplantısı"},
        ]

    results = fetch_latest_disclosures("ASELS.IS", limit=5, custom_fetcher=mock_fetcher)
    assert len(results) == 2
    assert results[0].ticker == "ASELS"
    assert results[0].category == EventCategory.POSITIVE


def test_fetch_latest_disclosures_network_error_graceful_fallback():
    def faulty_fetcher(ticker, limit):
        raise ConnectionError("Ağ bağlantısı koptu")

    results = fetch_latest_disclosures("EREGL.IS", limit=5, custom_fetcher=faulty_fetcher)
    assert results == []
