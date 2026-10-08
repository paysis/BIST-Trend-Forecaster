# tests/test_inference.py - canlı tahmin (app.py) senaryosunu taklit eden birim testleri
import os

import pandas as pd
import pytest
import xgboost as xgb

import config

FEATURES = config.FEATURES

pytestmark = pytest.mark.skipif(not os.path.exists(config.MODEL_PATH), reason="model dosyası yok")


def test_shipped_model_predicts_on_single_row_dataframe():
    """app.py her çalıştığında modele TEK satırlık bir DataFrame verir (günün son
    tahmini). xgboost'un pandas sütun adlarını tanıyamadığı bir ortamda (örn.
    pandas>=3 ile uyumsuz bir xgboost sürümü) bu çağrı "data did not contain
    feature names" hatasıyla patlar ve arayüzde her tahminde hata gösterilir."""
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)

    single_row = pd.DataFrame([{name: 0.0 for name in FEATURES}])
    proba = model.predict_proba(single_row)

    assert proba.shape == (1, 2)
    assert ((proba >= 0) & (proba <= 1)).all()
