# src/model_train.py
import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
import config
import features
import metrics
import tune as tune_module


def get_temporal_split(df_processed, features_list, train_ratio=0.9):
    """
    Veriyi benzersiz tarihlere göre (walk-forward mantığıyla) eğitim ve test setlerine ayırır.
    
    Tüm hisselerin belirli bir tarihten önceki verilerini eğitim, o tarih ve sonrasındaki
    verilerini ise test seti yapar. Böylece 'look-ahead bias' ve hisse bazlı veri sızıntısı
    (data leakage) engellenir.
    """
    unique_dates = np.sort(df_processed['Date'].unique())
    if len(unique_dates) < 2:
        raise ValueError("Zaman serisi ayrımı (temporal split) için en az 2 farklı tarih gereklidir.")

    cutoff_index = max(1, min(int(len(unique_dates) * train_ratio), len(unique_dates) - 1))
    cutoff_date = unique_dates[cutoff_index]

    train_mask = df_processed['Date'] < cutoff_date
    test_mask = df_processed['Date'] >= cutoff_date

    X_train = df_processed.loc[train_mask, features_list].reset_index(drop=True)
    X_test = df_processed.loc[test_mask, features_list].reset_index(drop=True)
    y_train = df_processed.loc[train_mask, 'target'].reset_index(drop=True)
    y_test = df_processed.loc[test_mask, 'target'].reset_index(drop=True)
    dates_train = df_processed.loc[train_mask, 'Date'].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, dates_train, cutoff_date


def train_model(tune=False, n_trials=50):
    # 1. Veriyi Yükle
    print("Veri yükleniyor...")
    df = pd.read_csv(config.DATA_PATH)
    
    # 2. Feature Engineering
    print("Feature Engineering uygulanıyor...")
    df_processed = features.add_features(df)
    
    # 3. Eğitim Seti Hazırlığı
    # Geleceği görmeyi engellemek için tarihsel kesim yapıyoruz (TimeSeries Split mantığı)
    features_list = ['rsi', 'macd', 'sma_10', 'sma_50', 'bb_width', 
                     'volatility', 'lag_1_ret', 'lag_2_ret', 'vol_change', 
                     'day_of_week', 'month']
    
    X_train, X_test, y_train, y_test, dates_train, cutoff_date = get_temporal_split(
        df_processed, features_list, train_ratio=0.9
    )
    
    cutoff_str = pd.to_datetime(cutoff_date).strftime('%Y-%m-%d')
    train_start = pd.to_datetime(dates_train.min()).strftime('%Y-%m-%d')
    train_end = pd.to_datetime(dates_train.max()).strftime('%Y-%m-%d')
    test_dates = df_processed.loc[df_processed['Date'] >= cutoff_date, 'Date']
    test_start = pd.to_datetime(test_dates.min()).strftime('%Y-%m-%d')
    test_end = pd.to_datetime(test_dates.max()).strftime('%Y-%m-%d')
    
    print(f"Tarih Kesimi (Cutoff): {cutoff_str}")
    print(f"Eğitim Verisi: {X_train.shape} (Tarih Aralığı: {train_start} - {train_end})")
    print(f"Test Verisi: {X_test.shape} (Tarih Aralığı: {test_start} - {test_end})")
    for name, y_part in (("Eğitim", y_train), ("Test", y_test)):
        dist = metrics.class_distribution(y_part)
        print(f"{name} Sınıf Dağılımı: Yükseliş %{dist[1] * 100:.1f} / Düşüş %{dist[0] * 100:.1f}")
    
    # 4. Model Tanımlama ve Eğitim (XGBoost)
    # Varsayılan: manuel parametreler. --tune ile Optuna optimizasyonu yapılır.
    params = dict(tune_module.DEFAULT_PARAMS)
    if tune:
        print(f"Optuna ile hiperparametre optimizasyonu ({n_trials} deneme)...")
        study = tune_module.run_study(X_train, y_train, dates_train, n_trials=n_trials)
        print(f"En iyi CV doğruluğu: {study.best_value:.4f}")
        print(f"En iyi parametreler: {study.best_params}")
        params = study.best_params

    model = xgb.XGBClassifier(**params, **tune_module.FIXED_PARAMS)
    
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)
    
    # 5. Değerlendirme
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    scores = metrics.evaluate(y_test, proba)
    acc = scores['accuracy']
    print(f"\n🎯 Model Doğruluğu (Test Seti): {acc:.4f}")
    print(f"Dengeli Doğruluk (Balanced Accuracy): {scores['balanced_accuracy']:.4f}")
    print(f"ROC-AUC: {scores['roc_auc']:.4f}")
    print(f"Log Loss: {scores['log_loss']:.4f}")
    print(f"Kesinlik (Precision, Yükseliş): {scores['precision']:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, preds))
    
    # Feature Importance (PDF Maddesi: Model Evaluation)
    importance = dict(zip(features_list, model.feature_importances_))
    print("\nÖnem Düzeyleri:")
    for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.4f}")
    
    # 6. Modeli Kaydet
    if not os.path.exists(os.path.dirname(config.MODEL_PATH)):
        os.makedirs(os.path.dirname(config.MODEL_PATH))
        
    model.save_model(config.MODEL_PATH)
    print(f"\n✅ Model kaydedildi: {config.MODEL_PATH}")
    return model, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIST XGBoost model eğitimi")
    parser.add_argument("--tune", action="store_true", help="Optuna ile hiperparametre optimizasyonu yap")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna deneme sayısı")
    args = parser.parse_args()
    train_model(tune=args.tune, n_trials=args.n_trials)