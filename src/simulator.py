# src/simulator.py
"""BIST 30 TL Bazlı What-If Yatırım ve Portföy Büyüme Simülatörü (#52).

Kullanıcının belirlediği başlangıç sermayesi (örn: 100.000 TL), hisse senedi ve model
sinyalleri doğrultusunda portföyün zaman içindeki TL bazlı büyümesini, Al ve Tut
(Buy & Hold) stratejisi ile karşılaştırmalı olarak hesaplar.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    """Yatırım simülasyonu özet metrikleri ve günlük portföy serisi."""

    initial_capital: float
    final_value: float
    net_profit_tl: float
    total_return_pct: float
    buy_hold_final_value: float
    buy_hold_net_profit_tl: float
    buy_hold_return_pct: float
    alpha_tl: float
    total_commission_paid: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_pct: float
    daily_timeline: pd.DataFrame


def simulate_investment(
    frame: pd.DataFrame,
    probs: Sequence[float],
    threshold: float = 0.53,
    initial_capital: float = 100_000.0,
    cost: float = 0.001,
) -> Optional[SimulationResult]:
    """Model sinyallerine göre TL bazlı portföy yatırım simülasyonunu çalıştırır.

    Parametreler:
        frame: 'Date' ve 'close' sütunlarını içeren fiyat DataFrame'i.
        probs: Modelin her gün için ürettiği yükseliş olasılıkları dizisi.
        threshold: Pozisyona giriş için gereken asgari olasılık eşiği (örn: 0.53).
        initial_capital: Başlangıç nakit sermayesi (TL, varsayılan 100.000 TL).
        cost: İşlem başına komisyon ve kayma oranı (varsayılan: 0.001 / %0.10).
    """
    if initial_capital <= 0:
        raise ValueError("Başlangıç sermayesi sıfırdan büyük olmalıdır.")

    if frame is None or len(frame) == 0:
        return None

    if len(probs) != len(frame):
        raise ValueError("Olasılık dizisinin uzunluğu ile DataFrame satır sayısı eşit olmalıdır.")

    closes = frame["close"].to_numpy(dtype=float)
    dates = frame["Date"].to_numpy()
    n = len(closes)

    if n < 1:
        return None

    probs_arr = np.asarray(probs, dtype=float)
    signals = (probs_arr >= threshold).astype(int)

    # --- Model Stratejisi Simülasyonu ---
    cash = float(initial_capital)
    shares = 0.0
    position = 0
    total_commission = 0.0
    trades_count = 0

    timeline_records = []
    trade_entry_price = 0.0
    winning_trades = 0
    losing_trades = 0

    for i in range(n):
        curr_price = float(closes[i])
        curr_date = dates[i]
        desired_pos = int(signals[i])

        # Pozisyon Değişikliği Mantığı
        if desired_pos == 1 and position == 0:
            # ALIM: Eldeki tüm nakit ile hisse al
            comm = cash * cost
            cash_after_comm = cash - comm
            shares = cash_after_comm / curr_price
            cash = 0.0
            position = 1
            total_commission += comm
            trades_count += 1
            trade_entry_price = curr_price

        elif desired_pos == 0 and position == 1:
            # SATIM / NAKDE GEÇİŞ: Hisseleri satıp nakde geç
            gross_proceeds = shares * curr_price
            comm = gross_proceeds * cost
            cash = gross_proceeds - comm
            shares = 0.0
            position = 0
            total_commission += comm
            trades_count += 1

            if curr_price > trade_entry_price:
                winning_trades += 1
            elif curr_price < trade_entry_price:
                losing_trades += 1

        stock_val = shares * curr_price
        total_port_val = cash + stock_val

        timeline_records.append(
            {
                "Date": curr_date,
                "close": curr_price,
                "signal": desired_pos,
                "position": position,
                "cash": cash,
                "shares": shares,
                "stock_value": stock_val,
                "portfolio_value": total_port_val,
            }
        )

    # Simülasyon sonunda hâlâ hissede isek son trade'i değerlendir
    if position == 1 and trades_count > 0:
        if closes[-1] > trade_entry_price:
            winning_trades += 1
        elif closes[-1] < trade_entry_price:
            losing_trades += 1

    timeline_df = pd.DataFrame(timeline_records)

    # --- Al ve Tut (Buy & Hold) Simülasyonu ---
    bh_comm_entry = initial_capital * cost
    bh_shares = (initial_capital - bh_comm_entry) / closes[0]
    bh_values = bh_shares * closes
    timeline_df["buy_hold_value"] = bh_values

    # Özet Metrikler
    final_val = float(timeline_df["portfolio_value"].iloc[-1])
    net_profit = final_val - initial_capital
    return_pct = (final_val / initial_capital - 1.0) * 100.0

    bh_final = float(bh_values[-1])
    bh_net_profit = bh_final - initial_capital
    bh_return_pct = (bh_final / initial_capital - 1.0) * 100.0
    alpha = final_val - bh_final

    # Maksimum Değer Kaybı (Max Drawdown %)
    running_max = timeline_df["portfolio_value"].cummax()
    drawdowns = (timeline_df["portfolio_value"] - running_max) / running_max
    max_dd_pct = float(drawdowns.min() * 100.0)

    total_evaluated_trades = winning_trades + losing_trades
    win_rate_val = (
        float(winning_trades / total_evaluated_trades) if total_evaluated_trades > 0 else 0.0
    )

    return SimulationResult(
        initial_capital=round(initial_capital, 2),
        final_value=round(final_val, 2),
        net_profit_tl=round(net_profit, 2),
        total_return_pct=round(return_pct, 2),
        buy_hold_final_value=round(bh_final, 2),
        buy_hold_net_profit_tl=round(bh_net_profit, 2),
        buy_hold_return_pct=round(bh_return_pct, 2),
        alpha_tl=round(alpha, 2),
        total_commission_paid=round(total_commission, 2),
        trades_count=trades_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=round(win_rate_val, 4),
        max_drawdown_pct=round(max_dd_pct, 2),
        daily_timeline=timeline_df,
    )
