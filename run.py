"""Reproduce the project's results.

1. Cross-validate the SMOTE + gradient boosting model on the training data
   to choose the probability threshold that maximises NIR.
2. Train on the full training data and score the test set once.

Usage:  python run.py
"""
import numpy as np

from promotion import (BENCHMARK_IRR, BENCHMARK_NIR, cross_validated_scores, fit_on_promoted,
                       load_data, make_model, promotion_strategy, score, test_results)

THRESHOLDS = np.round(np.arange(0.30, 0.801, 0.05), 2)


def main():
    train, test = load_data()

    print("Choosing the threshold by 5-fold cross-validation on the training data...")
    probs = cross_validated_scores(train, {"SMOTE + GB": make_model})["SMOTE + GB"]
    cv_nir = [score(train[probs > t])[1] * len(test) / len(train) for t in THRESHOLDS]
    for t, nir in zip(THRESHOLDS, cv_nir):
        print(f"  threshold {t:.2f}: NIR {nir:7.1f}")
    threshold = THRESHOLDS[int(np.argmax(cv_nir))]
    print(f"Chosen threshold: {threshold:.2f}\n")

    model = fit_on_promoted(make_model(), train)
    strategy = promotion_strategy(model, threshold)
    irr, nir = test_results(strategy, test)
    promoted = (strategy(test) == "Yes").mean()
    all_irr, all_nir = score(test)

    print(f"Test set results ({len(test):,} customers)")
    print(f"  {'Strategy':30s} {'Promoted':>9s} {'IRR':>8s} {'NIR':>10s}")
    print(f"  {'Promote to everyone':30s} {1:9.0%} {all_irr:8.4f} {all_nir:10.2f}")
    print(f"  {'Starbucks benchmark':30s} {'-':>9s} {BENCHMARK_IRR:8.4f} {BENCHMARK_NIR:10.2f}")
    print(f"  {'SMOTE + gradient boosting':30s} {promoted:9.0%} {irr:8.4f} {nir:10.2f}")


if __name__ == "__main__":
    main()
