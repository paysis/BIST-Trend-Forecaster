# tests/test_integration.py - gerçek veri seti ile uçtan uca eğitim/optimizasyon
import os

import pandas as pd
import pytest
import xgboost as xgb

import config
import features
import model_train
import tune

FEATURES = ['rsi', 'macd', 'sma_10', 'sma_50', 'bb_width',
            'volatility', 'lag_1_ret', 'lag_2_ret', 'vol_change',
            'day_of_week', 'month']

pytestmark = pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="veri seti yok")


@pytest.fixture
def small_dataset(tmp_path, monkeypatch):
    """Gerçek CSV'den 3 hisse x ~1 yıllık küçük bir kesit; config yolları tmp'ye yönlendirilir."""
    df = pd.read_csv(config.DATA_PATH)
    df = df[df["ticker"].isin(["AKBNK", "GARAN", "THYAO"]) & (df["Date"] >= "2024-01-01")]
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    monkeypatch.setattr(config, "DATA_PATH", str(data_path))
    monkeypatch.setattr(config, "MODEL_PATH", str(tmp_path / "models" / "model.json"))
    return df


def test_features_and_study_on_real_data(small_dataset):
    processed = features.add_features(small_dataset)
    study = tune.run_study(processed[FEATURES], processed["target"], processed["Date"],
                           n_trials=2, n_splits=2)
    assert 0.0 <= study.best_value <= 1.0


@pytest.mark.parametrize("use_tune", [False, True])
def test_train_model_saves_loadable_model(small_dataset, use_tune):
    model, acc = model_train.train_model(tune=use_tune, n_trials=2)
    assert 0.0 <= acc <= 1.0
    assert os.path.exists(config.MODEL_PATH)

    # app.py ile aynı şekilde yükle ve tahmin yap
    loaded = xgb.XGBClassifier()
    loaded.load_model(config.MODEL_PATH)
    X = features.add_features(small_dataset)[FEATURES].tail(5)
    proba = loaded.predict_proba(X)
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_shipped_model_still_loads():
    """Repodaki modelin eğitim kodundaki değişiklikten etkilenmediğini doğrular."""
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    assert model.n_features_in_ == len(FEATURES)


def test_train_model_reports_imbalance_aware_metrics(small_dataset, capsys):
    model_train.train_model()
    out = capsys.readouterr().out
    assert "Eğitim Sınıf Dağılımı" in out
    assert "Test Sınıf Dağılımı" in out
    for label in ("Dengeli Doğruluk", "ROC-AUC", "Log Loss"):
        assert label in out
