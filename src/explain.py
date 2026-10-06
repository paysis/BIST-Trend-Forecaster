# src/explain.py - XGBoost TreeSHAP katkılarıyla tek bir tahminin yerel açıklaması
import numpy as np
import pandas as pd
import xgboost as xgb


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def explain_prediction(model, X_row):
    """Tek satırlık bir tahmini öznitelik katkılarına ayırır.

    XGBoost'un yerleşik TreeSHAP'i (pred_contribs) log-odds uzayında
    bias + sum(katkılar) == model çıktısı eşitliğini sağlar.

    Dönüş: (explanation, base_prob)
      explanation: |katkı|'ya göre azalan sıralı DataFrame
        - feature: öznitelik adı
        - value: öznitelik değeri
        - contribution: log-odds katkısı (SHAP değeri)
        - impact: olasılık uzayındaki etki; en büyük katkıdan başlayarak
          sırayla uygulanır, böylece base_prob + sum(impact) == tahmin olasılığı
      base_prob: hiçbir öznitelik bilinmediğinde modelin beklenen olasılığı
    """
    if len(X_row) != 1:
        raise ValueError(f"Tek bir satır bekleniyordu, {len(X_row)} satır verildi.")

    contribs = model.get_booster().predict(xgb.DMatrix(X_row), pred_contribs=True)[0]
    bias = contribs[-1]

    explanation = pd.DataFrame({
        "feature": list(X_row.columns),
        "value": X_row.iloc[0].to_numpy(dtype=float),
        "contribution": contribs[:-1],
    })
    order = explanation["contribution"].abs().sort_values(ascending=False, kind="stable").index
    explanation = explanation.loc[order].reset_index(drop=True)

    cumulative = bias + explanation["contribution"].cumsum().to_numpy()
    previous = np.concatenate([[bias], cumulative[:-1]])
    explanation["impact"] = _sigmoid(cumulative) - _sigmoid(previous)

    return explanation, float(_sigmoid(bias))
