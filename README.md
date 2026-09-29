# Targeted Promotion Optimisation with A/B Testing

Starbucks ran an A/B test on about 126,000 customers. Half were sent a promotion for a \$10 product and half were not. Each promotion costs \$0.15 to send, and although it increased purchases, sending it to everyone lost money.

This project:

1. uses **hypothesis testing** to validate the A/B test and measure the promotion's effect;
2. trains a **gradient boosting classifier**, with **SMOTE over-sampling** to handle class imbalance, to predict which customers respond to the promotion;
3. sends the promotion only to those customers, turning a loss into a profit.

## Results

On the held-out test set of 41,650 customers:

| Strategy | Customers promoted | IRR | NIR |
|---|---:|---:|---:|
| Promote to everyone | 100% | 0.0096 | −\$1,132 |
| Starbucks' own model (benchmark) | – | 0.0188 | \$189 |
| **This project** | **28%** | **0.0207** | **\$320** |

- **IRR (Incremental Response Rate):** how much the promotion raises the purchase rate among the customers promoted.
- **NIR (Net Incremental Revenue):** the extra revenue the promotion creates, minus the cost of sending it.

Targeting turns a \$1,132 loss into a \$320 profit, about 1.7 times the Starbucks benchmark.

## Method

### 1. A/B test analysis

- **Randomisation check:** a binomial test confirms the 50/50 split (p = 0.51), and the two groups are balanced on every feature.
- **Hypothesis test:** a one-sided two-proportion z-test shows the promotion significantly increased the purchase rate, from 0.76% to 1.70% (z = 12.47, p < 0.001). The 95% confidence interval for the increase is 0.80 to 1.09 percentage points.
- **Power analysis:** 762 customers per group would be enough to detect the break-even effect with 80% power. The experiment has about 42,000 per group, so it is very well powered.
- **Profitability:** each extra sale earns \$10 and each promotion costs \$0.15, so the promotion only pays for itself if it raises the purchase rate by at least 1.5 points. The whole confidence interval is below that, so promoting everyone loses money.

### 2. Class imbalance

Purchases are rare: among promoted customers, the data the model learns from, there are 58 non-buyers for every buyer. A classifier trained on this data as it is predicts that nobody will buy. It is 98.3% accurate but useless for targeting.

**SMOTE-NC** fixes this by creating synthetic buyers, interpolating between similar real ones until the classes are balanced. The NC version handles the mix of categorical and continuous features. It runs inside an `imblearn` pipeline, so it is only applied to training data, never to the customers being scored.

### 3. Modelling

- **Model:** a gradient boosting classifier (scikit-learn's `HistGradientBoostingClassifier`), trained on promoted customers to predict each customer's chance of buying when promoted.
- **Threshold:** customers above a probability threshold get the promotion. The threshold is chosen by 5-fold cross-validation on the training data to maximise NIR. The best was the standard 0.5, because SMOTE trained the model on balanced classes.
- **Evaluation:** the test set is used once, only after the model and threshold are fixed.
- **Key features:** `V4` and `V5` drive the model's decisions. Customers with `V4 = 1` don't respond to the promotion at all.

## How to run

```bash
git clone <your-repo-url>
cd Targeted-Promotion-Optimisation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To reproduce the results table (takes about 30 seconds):

```bash
python run.py
```

To open the full analysis, with charts and explanations of every step:

```bash
jupyter notebook Starbucks_Promotion_Optimisation.ipynb
```

## Limitations and next steps

- **The model finds customers who buy when promoted, not customers who buy *because* of the promotion.** Some would have bought anyway. Uplift modelling, such as a T-learner that compares models trained on promoted and control customers, targets the extra buyers directly and is the natural next step.
- **SMOTE inflates the predicted probabilities.** The model's average is about 35% against a real purchase rate of 1.7%, so the scores work for ranking customers but aren't true probabilities. Class weights are an alternative that avoids synthetic data.
- **The features are anonymised**, so the customer groups the model selects can't be described in business terms.
