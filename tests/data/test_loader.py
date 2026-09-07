"""Tests for the price data loader: caching correctness, not network access."""

from __future__ import annotations

import pandas as pd

from ml_backtester.config import Config
from ml_backtester.data import loader


def _fake_ohlcv(start: str, end: str) -> pd.DataFrame:
    """Build a small deterministic OHLCV frame over a business-day date range."""
    index = pd.date_range(start, end, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": range(len(index)),
            "high": range(len(index)),
            "low": range(len(index)),
            "close": range(len(index)),
            "volume": [1000] * len(index),
        },
        index=index,
    )


def test_load_ticker_downloads_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return _fake_ohlcv(start, end)

    monkeypatch.setattr(loader, "_download_ticker", fake_download)

    data = loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path))

    assert len(calls) == 1
    assert not data.empty
    assert loader._cache_path(str(tmp_path), "AAPL").exists()


def test_load_ticker_reuses_cache_without_redownload(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        loader, "_download_ticker", lambda t, s, e: calls.append(1) or _fake_ohlcv(s, e)
    )

    loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path))
    loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path))

    assert len(calls) == 1


def test_load_ticker_force_refresh_redownloads(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        loader, "_download_ticker", lambda t, s, e: calls.append(1) or _fake_ohlcv(s, e)
    )

    loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path))
    loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path), force_refresh=True)

    assert len(calls) == 2


def test_load_ticker_widening_range_triggers_redownload(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        loader, "_download_ticker", lambda t, s, e: calls.append((s, e)) or _fake_ohlcv(s, e)
    )

    loader.load_ticker("AAPL", "2020-01-15", "2020-01-31", str(tmp_path))
    data = loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path))

    assert len(calls) == 2
    assert data.index.min() <= pd.Timestamp("2020-01-01")


def test_load_universe_combines_all_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_download_ticker", lambda t, s, e: _fake_ohlcv(s, e))

    config = Config(
        universe=["AAPL", "MSFT"],
        start_date="2020-01-01",
        end_date="2020-01-10",
        cache_dir=str(tmp_path),
    )
    data = loader.load_universe(config)

    assert set(data["ticker"].unique()) == {"AAPL", "MSFT"}
    assert data.index.is_monotonic_increasing
    assert list(data.columns) == ["open", "high", "low", "close", "volume", "ticker"]


def test_load_ticker_respects_requested_date_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_download_ticker", lambda t, s, e: _fake_ohlcv(s, e))

    loader.load_ticker("AAPL", "2020-01-01", "2020-01-31", str(tmp_path))
    narrower = loader.load_ticker("AAPL", "2020-01-10", "2020-01-15", str(tmp_path))

    assert narrower.index.min() >= pd.Timestamp("2020-01-10")
    assert narrower.index.max() <= pd.Timestamp("2020-01-15")
