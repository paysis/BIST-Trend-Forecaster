# tests/test_data_ingestion.py - toplu veri çekme (data_ingestion.py) testleri
import pandas as pd
import pytest
import yfinance

import config
import data_ingestion


@pytest.fixture
def two_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TICKERS", ["AKBNK.IS", "GARAN.IS"])
    monkeypatch.setattr(config, "MACRO_TICKERS", [])
    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path / "data" / "bist.csv"))


def _frame(*args, **kwargs):
    dates = pd.bdate_range("2025-01-01", periods=3)
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100.0},
        index=pd.Index(dates, name="Date"),
    )


def test_fetch_data_recovers_from_transient_errors(two_tickers, monkeypatch):
    failures = {"AKBNK.IS": 2}

    def download(ticker, *args, **kwargs):
        if failures.get(ticker, 0) > 0:
            failures[ticker] -= 1
            raise ConnectionError("ağ hatası")
        return _frame()

    monkeypatch.setattr(yfinance, "download", download)

    data_ingestion.fetch_data()

    saved = pd.read_csv(config.DATA_PATH)
    assert sorted(saved["ticker"].unique()) == ["AKBNK", "GARAN"]
    assert len(saved) == 6


def test_fetch_data_skips_ticker_that_stays_unreachable(two_tickers, monkeypatch, capsys):
    def download(ticker, *args, **kwargs):
        if ticker == "AKBNK.IS":
            raise ConnectionError("ağ hatası")
        return _frame()

    monkeypatch.setattr(yfinance, "download", download)

    data_ingestion.fetch_data()

    saved = pd.read_csv(config.DATA_PATH)
    assert list(saved["ticker"].unique()) == ["GARAN"]
    assert "AKBNK.IS hatası" in capsys.readouterr().out


def test_fetch_data_downloads_macro_tickers(two_tickers, monkeypatch):
    monkeypatch.setattr(config, "MACRO_TICKERS", ["XU100.IS", "USDTRY=X"])
    monkeypatch.setattr(yfinance, "download", lambda ticker, *args, **kwargs: _frame())

    data_ingestion.fetch_data(include_macro=True)

    saved = pd.read_csv(config.DATA_PATH)
    assert set(saved["ticker"].unique()) == {"AKBNK", "GARAN", "XU100", "USDTRY"}
    assert len(saved) == 12


def test_fetch_incremental_falls_back_to_full_when_file_missing(two_tickers, monkeypatch):
    monkeypatch.setattr(yfinance, "download", lambda ticker, *args, **kwargs: _frame())

    # DATA_PATH mevcut değilken fetch_incremental çağrılır
    df = data_ingestion.fetch_incremental()
    assert len(df) == 6
    assert set(df["ticker"].unique()) == {"AKBNK", "GARAN"}


def test_fetch_incremental_appends_only_new_dates_without_duplicates(two_tickers, monkeypatch):
    # Mevcut veriyi oluştur (2025-01-01 - 2025-01-03)
    existing_dates = pd.bdate_range("2025-01-01", periods=3)
    existing_rows = []
    for t in ["AKBNK", "GARAN"]:
        for d in existing_dates:
            existing_rows.append(
                {
                    "Date": d.strftime("%Y-%m-%d"),
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100.0,
                    "ticker": t,
                }
            )
    existing_df = pd.DataFrame(existing_rows)
    import os

    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    existing_df.to_csv(config.DATA_PATH, index=False)

    # Yeni veri indirme fonksiyonu (2025-01-04 - 2025-01-05 ve kısmi 2025-01-03 çakışması)
    download_calls = []

    def mock_download(ticker, start=None, end=None, **kwargs):
        download_calls.append((ticker, start, end))
        new_dates = pd.bdate_range(start or "2025-01-04", periods=2)
        return pd.DataFrame(
            {"Open": 2.0, "High": 3.0, "Low": 1.5, "Close": 2.5, "Volume": 200.0},
            index=pd.Index(new_dates, name="Date"),
        )

    monkeypatch.setattr(yfinance, "download", mock_download)

    # Incremental çalıştır
    result_df = data_ingestion.fetch_data(incremental=True, end_date="2025-01-10")

    # Start tarihi en son tarihten sonraki gün olmalı (2025-01-04)
    assert len(download_calls) == 2
    for _, start, _ in download_calls:
        assert start == "2025-01-04"

    # Mükerrer satır olmamalı
    assert not result_df.duplicated(subset=["Date", "ticker"]).any()
    # 3 eski + 2 yeni = 5 gün x 2 hisse = 10 satır
    assert len(result_df) == 10


def test_fetch_incremental_does_nothing_when_already_up_to_date(two_tickers, monkeypatch):
    import os

    existing = pd.DataFrame(
        [
            {
                "Date": "2025-05-10",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100.0,
                "ticker": "AKBNK",
            }
        ]
    )
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    existing.to_csv(config.DATA_PATH, index=False)

    downloaded = False

    def mock_download(*args, **kwargs):
        nonlocal downloaded
        downloaded = True
        return _frame()

    monkeypatch.setattr(yfinance, "download", mock_download)

    # end_date mevcuttan daha eski veya eşitse indirme yapılmaz
    result_df = data_ingestion.fetch_incremental(end_date="2025-05-10")
    assert not downloaded
    assert len(result_df) == 1
