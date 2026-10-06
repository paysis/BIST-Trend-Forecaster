# tests/test_explain.py - TreeSHAP tabanlı yerel açıklanabilirlik testleri
import numpy as np
import pytest
import xgboost as xgb

from explain import explain_prediction


@pytest.fixture
def trained_model(synthetic_data):
    X, y, _ = synthetic_data
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
    model.fit(X, y)
    return model, X


def test_contributions_cover_every_feature(trained_model):
    model, X = trained_model
    explanation, _ = explain_prediction(model, X.iloc[[0]])

    assert sorted(explanation["feature"]) == sorted(X.columns)
    assert list(explanation.columns) == ["feature", "value", "contribution", "impact"]


def test_contributions_sum_to_model_margin(trained_model):
    """TreeSHAP yerel doğruluk: bias + katkılar == modelin log-odds çıktısı."""
    model, X = trained_model
    row = X.iloc[[5]]
    explanation, base_prob = explain_prediction(model, row)

    margin = model.predict(row, output_margin=True)[0]
    base_margin = np.log(base_prob / (1 - base_prob))
    assert base_margin + explanation["contribution"].sum() == pytest.approx(margin, abs=1e-5)


def test_probability_impacts_reconcile_to_predicted_probability(trained_model):
    model, X = trained_model
    row = X.iloc[[7]]
    explanation, base_prob = explain_prediction(model, row)

    prob = model.predict_proba(row)[0][1]
    assert base_prob + explanation["impact"].sum() == pytest.approx(prob, abs=1e-5)


def test_impact_sign_follows_contribution_sign(trained_model):
    model, X = trained_model
    explanation, _ = explain_prediction(model, X.iloc[[3]])

    nonzero = explanation[explanation["contribution"] != 0]
    assert (np.sign(nonzero["impact"]) == np.sign(nonzero["contribution"])).all()


def test_sorted_by_absolute_contribution(trained_model):
    model, X = trained_model
    explanation, _ = explain_prediction(model, X.iloc[[0]])

    magnitudes = explanation["contribution"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_dominant_feature_ranks_first(trained_model):
    """Sentetik hedef f0'a bağlı; uç bir f0 değeri en büyük katkıyı vermeli."""
    model, X = trained_model
    row = X.iloc[[X["f0"].abs().idxmax()]]
    explanation, _ = explain_prediction(model, row)

    assert explanation.iloc[0]["feature"] == "f0"


def test_reports_feature_values_of_the_explained_row(trained_model):
    model, X = trained_model
    row = X.iloc[[2]]
    explanation, _ = explain_prediction(model, row)

    values = explanation.set_index("feature")["value"]
    for col in X.columns:
        assert values[col] == pytest.approx(row[col].iloc[0])


def test_rejects_multiple_rows(trained_model):
    model, X = trained_model
    with pytest.raises(ValueError):
        explain_prediction(model, X.iloc[:2])
