# src/tune.py
# Optuna ile XGBoost hiperparametre optimizasyonu
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score

# model_train.py'deki manuel parametreler: optimizasyon bu noktadan başlar
DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 5,
}

# XGBoost'a her denemede sabit geçilen parametreler
FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_jobs': -1,
}


def suggest_params(trial):
    """Bir Optuna denemesi için XGBoost hiperparametre önerir."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }


def time_series_folds(dates, n_splits=3):
    """
    Tarihe göre ileri doğru (walk-forward) CV katları üretir.
    Aynı günün satırları (farklı hisseler) hep aynı katta kalır, böylece
    doğrulama seti her zaman eğitim setinden sonraki günleri içerir.
    """
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    if len(unique_dates) < n_splits + 1:
        raise ValueError(f"{n_splits} kat için en az {n_splits + 1} farklı gün gerekli.")

    fold_size = len(unique_dates) // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = unique_dates[i * fold_size]
        val_end = unique_dates[(i + 1) * fold_size] if i < n_splits else None
        train_idx = np.where(dates < train_end)[0]
        if val_end is None:
            val_idx = np.where(dates >= train_end)[0]
        else:
            val_idx = np.where((dates >= train_end) & (dates < val_end))[0]
        yield train_idx, val_idx


def make_objective(X, y, dates, n_splits=3):
    """Zaman serisi CV'sindeki ortalama doğruluğu maksimize eden objective fonksiyonu."""
    folds = list(time_series_folds(dates, n_splits))

    def objective(trial):
        params = suggest_params(trial)
        scores = []
        for step, (train_idx, val_idx) in enumerate(folds):
            model = xgb.XGBClassifier(**params, **FIXED_PARAMS)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            scores.append(accuracy_score(y.iloc[val_idx], preds))

            # Kötü giden denemeleri erkenden buda
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def run_study(X, y, dates, n_trials=50, n_splits=3, seed=42, timeout=None):
    """
    Optuna çalışmasını başlatır ve tamamlanmış study nesnesini döndürür.
    En iyi parametreler: study.best_params
    """
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    # Mevcut manuel parametreleri ilk deneme olarak ekle (baseline)
    study.enqueue_trial(DEFAULT_PARAMS)
    study.optimize(make_objective(X, y, dates, n_splits), n_trials=n_trials, timeout=timeout)
    return study
