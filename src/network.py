# src/network.py - Yahoo Finance istekleri için üstel geri çekilmeli (exponential backoff) retry
import time

import yfinance as yf

# Testlerde gerçek beklemeyi atlamak için değiştirilebilir
_sleep = time.sleep


class MarketDataUnavailableError(ValueError):
    """Tüm denemelere rağmen Yahoo Finance'e ulaşılamadığında fırlatılır."""


def download_with_retry(*args, attempts=3, min_delay=2, max_delay=10, **kwargs):
    """
    yf.download'ı geçici hatalara karşı yeniden deneyerek çağırır.

    yfinance ağ ve hız sınırı (HTTP 429) hatalarının çoğunu yutup boş DataFrame
    döndürdüğü için hem istisnalar hem de boş sonuçlar yeniden denenir.
    Denemeler arasındaki gecikme min_delay'den başlayıp her seferinde ikiye
    katlanır ve max_delay ile sınırlanır (2s, 4s, 8s, 10s...).

    Son deneme de boş dönerse boş sonuç döndürülür (örn. geçersiz sembol);
    hepsi istisna fırlatırsa MarketDataUnavailableError fırlatılır.
    """
    last_error = None
    result = None
    for attempt in range(attempts):
        if attempt > 0:
            _sleep(min(max_delay, min_delay * 2 ** (attempt - 1)))
        try:
            result = yf.download(*args, **kwargs)
        except Exception as e:
            last_error = e
            continue
        last_error = None
        if result is not None and not result.empty:
            return result

    if last_error is not None:
        raise MarketDataUnavailableError(
            f"Yahoo Finance'e {attempts} denemede ulaşılamadı. Geçici bir bağlantı sorunu "
            "olabilir; lütfen birkaç dakika sonra tekrar deneyin."
        ) from last_error
    return result
