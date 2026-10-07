# src/signals.py - model olasılığını 3 durumlu yön sinyaline dönüştürür

UP = "up"
NEUTRAL = "neutral"
DOWN = "down"


def classify_signal(prob, low, high):
    """Artış olasılığını yön sinyaline çevirir.

    prob >= high -> UP, prob <= low -> DOWN, aradaki değerler -> NEUTRAL.
    Eşikler dahildir; low < high olmalıdır.
    """
    if not low < high:
        raise ValueError(f"Alt eşik ({low}) üst eşikten ({high}) küçük olmalı.")
    if prob >= high:
        return UP
    if prob <= low:
        return DOWN
    return NEUTRAL


def symmetric_band(min_confidence):
    """Minimum güven eşiğinden 0.5 etrafında simetrik (low, high) bandı üretir.

    Örn: 0.55 -> (0.45, 0.55)
    """
    if not 0.5 < min_confidence < 1.0:
        raise ValueError(f"Minimum güven eşiği 0.5 ile 1.0 arasında olmalı: {min_confidence}")
    return round(1.0 - min_confidence, 10), min_confidence
