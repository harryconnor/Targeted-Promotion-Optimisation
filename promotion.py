"""Helper functions for the Starbucks promotion-targeting project.

The model is a response model: a gradient boosting classifier, trained on the
customers who received the promotion, that predicts how likely each customer
is to buy when promoted. SMOTE-NC over-sampling handles the class imbalance
(far fewer buyers than non-buyers).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES = [f"V{i}" for i in range(1, 8)]
CATEGORICAL = ["V1", "V4", "V5", "V6", "V7"]   # V2 and V3 are continuous

REVENUE_PER_PURCHASE = 10.0
COST_PER_PROMOTION = 0.15
BENCHMARK_IRR, BENCHMARK_NIR = 0.0188, 189.45   # Starbucks' own model on the test set


def load_data():
    """Return the (train, test) DataFrames."""
    return pd.read_csv(DATA_DIR / "training.csv"), pd.read_csv(DATA_DIR / "test.csv")


def score(targeted):
    """IRR and NIR for the customers a strategy chose to promote (Starbucks' definitions).

    Because the original experiment was randomised, about half of any targeted
    group actually received the promotion, so comparing them with the other
    half measures the promotion's effect on that group.
    """
    treated = targeted["Promotion"] == "Yes"
    n_treat, n_ctrl = treated.sum(), (~treated).sum()
    purch_treat = targeted.loc[treated, "purchase"].sum()
    purch_ctrl = targeted.loc[~treated, "purchase"].sum()
    irr = purch_treat / n_treat - purch_ctrl / n_ctrl if n_treat and n_ctrl else np.nan
    nir = (REVENUE_PER_PURCHASE * purch_treat
           - COST_PER_PROMOTION * n_treat
           - REVENUE_PER_PURCHASE * purch_ctrl)
    return irr, nir


def make_model(oversample=True, random_state=0):
    """Gradient boosting classifier, optionally preceded by SMOTE-NC over-sampling.

    SMOTE-NC is the version of SMOTE for data that mixes categorical and
    continuous features: it creates synthetic buyers by interpolating the
    continuous features between real buyers and copying the most common
    category among their neighbours. Inside an imblearn pipeline, over-sampling
    is applied only when the model is fitted, never to data being scored.
    """
    classifier = HistGradientBoostingClassifier(
        categorical_features=CATEGORICAL,
        learning_rate=0.05, max_iter=200, max_depth=3,   # shallow trees ...
        min_samples_leaf=200, l2_regularization=1.0,     # ... and large leaves to avoid overfitting
        random_state=random_state,
    )
    if not oversample:
        return classifier
    smote = SMOTENC(categorical_features=CATEGORICAL, random_state=random_state)
    return make_pipeline(smote, classifier)


def fit_on_promoted(model, df):
    """Train on promoted customers only: the target is 'did they buy when promoted?'."""
    promoted = df[df["Promotion"] == "Yes"]
    return model.fit(promoted[FEATURES], promoted["purchase"])


def cross_validated_scores(df, models, n_splits=5, random_state=0):
    """Out-of-fold purchase probabilities for every customer in `df`.

    Each fold's model is trained on the promoted customers in the other folds,
    then scores every customer (promoted and control) in the held-out fold.
    Returns {model name: array of probabilities}.
    """
    strata = df["Promotion"] + df["purchase"].astype(str)
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    scores = {name: np.zeros(len(df)) for name in models}
    for fit_idx, val_idx in folds.split(df, strata):
        for name, make in models.items():
            model = fit_on_promoted(make(), df.iloc[fit_idx])
            scores[name][val_idx] = model.predict_proba(df.iloc[val_idx][FEATURES])[:, 1]
    return scores


def promotion_strategy(model, threshold):
    """Starbucks interface: customer features -> array of 'Yes'/'No'."""
    def strategy(X):
        return np.where(model.predict_proba(X[FEATURES])[:, 1] > threshold, "Yes", "No")
    return strategy


def test_results(strategy, test):
    """Apply a strategy to the test set and return (IRR, NIR)."""
    promos = strategy(test[FEATURES])
    return score(test[promos == "Yes"])
