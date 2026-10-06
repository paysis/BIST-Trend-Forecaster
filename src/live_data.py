# src/live_data.py - Yahoo Finance'ten canlı veri çekip modele hazırlayan ortak yardımcılar
# (Tek hisse analizi ve BIST 30 taraması aynı işlem hattını kullanır.)
import pandas as pd
import yfinance as yf
from src import config, features

# Modelin indikatörleri hesaplayabilmesi için son 6 ayın verisine ihtiyacı var
LIVE_PERIOD = "6mo"


def yahoo_symbol(ticker):
    """Yahoo Finance sembolü değişen hisseleri eşler (Örn: KOZAL -> TRALT)."""
    return config.TICKER_YAHOO_MAP.get(ticker, ticker)


def prepare_live_frame(raw, ticker):
    """yf.download çıktısını features.add_features'ın beklediği biçime getirip
    indikatörleri hesaplar.

    Dönüş: (df_processed, df_ohlcv). df_processed'in son satırı en güncel işlem
    günüdür; df_ohlcv grafik ve fiyat metrikleri içindir.
    Veri yoksa veya indikatörler için yetersizse ValueError fırlatır.
    """
    if raw is None or raw.empty:
        raise ValueError(
            f"'{ticker}' (Yahoo: '{yahoo_symbol(ticker)}') için piyasa verisi alınamadı. "
            "Sembol değişmiş veya Yahoo Finance servisi yanıt vermiyor olabilir."
        )

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['ticker'] = ticker.replace(".IS", "")
    df.reset_index(inplace=True)

    # Sütun isimlerini düzenle (features.py 'Date' ve küçük harfli sütunlar bekliyor)
    new_columns = {}
    for col in df.columns:
        if col.lower() in ('date', 'index'):
            new_columns[col] = 'Date'
        elif col.lower() == 'ticker':
            new_columns[col] = 'ticker'
        else:
            new_columns[col] = col.lower()
    df.rename(columns=new_columns, inplace=True)

    if 'Date' not in df.columns:
        raise ValueError(f"'{ticker}' için çekilen veride 'Date' sütunu bulunamadı.")

    # drop_incomplete_target=False: canlı tahminde bugünün hedefi (yarının kapanışı)
    # henüz bilinmez; bu normalde eğitimde düşürülen son günü burada tutar.
    df_processed = features.add_features(df, drop_incomplete_target=False)
    if df_processed.empty:
        raise ValueError(f"'{ticker}' verisi teknik indikatörler hesaplandıktan sonra yetersiz kaldı.")

    return df_processed, df


def fetch_live_frame(ticker):
    """Tek bir hisse için canlı veriyi indirip prepare_live_frame ile hazırlar."""
    raw = yf.download(yahoo_symbol(ticker), period=LIVE_PERIOD, progress=False)
    return prepare_live_frame(raw, ticker)
