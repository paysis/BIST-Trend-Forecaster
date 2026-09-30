# src/model_train.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import config
import features
import tune as tune_module
import joblib
import os
import argparse

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
    
    X = df_processed[features_list]
    y = df_processed['target']
    
    # Son 3 ayı test verisi olarak ayıralım, gerisi eğitim
    split_point = int(len(df_processed) * 0.9)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"Eğitim Verisi: {X_train.shape}, Test Verisi: {X_test.shape}")
    
    # 4. Model Tanımlama ve Eğitim (XGBoost)
    # Varsayılan: manuel parametreler. --tune ile Optuna optimizasyonu yapılır.
    params = dict(tune_module.DEFAULT_PARAMS)
    if tune:
        print(f"Optuna ile hiperparametre optimizasyonu ({n_trials} deneme)...")
        dates_train = df_processed['Date'].iloc[:split_point]
        study = tune_module.run_study(X_train, y_train, dates_train, n_trials=n_trials)
        print(f"En iyi CV doğruluğu: {study.best_value:.4f}")
        print(f"En iyi parametreler: {study.best_params}")
        params = study.best_params

    model = xgb.XGBClassifier(**params, **tune_module.FIXED_PARAMS)
    
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)
    
    # 5. Değerlendirme
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 Model Doğruluğu (Test Seti): {acc:.4f}")
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