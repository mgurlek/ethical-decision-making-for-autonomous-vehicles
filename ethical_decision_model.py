"""Ethical decision model for autonomous vehicles.

Learns, from survey responses, how much each attribute of a potential victim
(age, health, species, number of people, ...) influences which side of a
dilemma respondents chose to save. A logistic regression is fitted on the
one-hot encoded scenarios and its coefficients are used as a scoring function
to pick between two candidate groups.
"""
import argparse
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = [
    'varlık', 'sağlık', 'yaş', 'ünlü', 'bağlılık', 'iyilik', 'zenginlik',
    'cinsiyet', 'uyruk', 'kilo', 'hamile', 'araç',
]
COUNT_COL = 'kişi_sayısı'
TARGET_COL = 'tercih_edildi'
ID_COL = 'katılımcı'
DROP_COLS = [TARGET_COL, 'soru', ID_COL]

DEFAULT_DATA = 'data/survey_long.csv'


def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    return encoded.drop(columns=DROP_COLS), encoded[TARGET_COL], df[ID_COL], df['soru']


def fit_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def puan(grup, coeffs, reference_columns):
    """Score a group described as {feature: value} using the learned weights."""
    score = coeffs.get(COUNT_COL, 0) * grup.get(COUNT_COL, 0)
    for feature, value in grup.items():
        if feature == COUNT_COL:
            continue
        col = f"{feature}_{value}"
        if col in reference_columns:
            score += coeffs.get(col, 0)
        else:
            warnings.warn(f"Unknown feature value ignored: {col}")
    return score


# Example scenarios taken from the survey design (Q12 and Q4)
ANNE = {**{k: 'belirtilmemiş' for k in CATEGORICAL_COLS}, 'varlık': 'insan',
        'bağlılık': 'yakın', 'hamile': 'hayır', 'araç': 'yaya', 'kişi_sayısı': 1}
BES_YABANCI = {**ANNE, 'bağlılık': 'yabancı', 'kişi_sayısı': 5}
BEBEK = {**ANNE, 'bağlılık': 'belirtilmemiş', 'yaş': 'bebek'}
YETISKIN = {**BEBEK, 'yaş': 'yetişkin'}


def pairwise_accuracy(model, X, y, respondent, question):
    """Share of (respondent, question) pairs where the spared group scores higher."""
    scores = pd.Series(model.decision_function(X), index=X.index)
    pair = pd.DataFrame({'s': scores, 'y': y, 'r': respondent, 'q': question})
    spared = pair[pair.y == 1].set_index(['r', 'q']).s
    hit = pair[pair.y == 0].set_index(['r', 'q']).s
    return (spared > hit).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=DEFAULT_DATA, help='survey CSV path')
    args = parser.parse_args()

    X, y, respondent, question = load_data(args.data)

    # Hold-out check of how well the model reproduces respondents' choices;
    # the split is by respondent so no one appears in both train and test
    train_ids, _ = train_test_split(respondent.unique(), test_size=0.2, random_state=42)
    is_train = respondent.isin(train_ids)
    X_train, X_test = X[is_train], X[~is_train]
    y_train, y_test = y[is_train], y[~is_train]
    holdout = fit_model(X_train, y_train)
    print(f"Hold-out accuracy (single group): {accuracy_score(y_test, holdout.predict(X_test)):.3f}")
    pair_acc = pairwise_accuracy(holdout, X_test, y_test, respondent[~is_train], question[~is_train])
    print(f"Hold-out accuracy (which group is spared): {pair_acc:.3f}")

    # Final model on all data
    model = fit_model(X, y)
    coefficients = pd.Series(model.coef_[0], index=X.columns)
    coeffs = coefficients.to_dict()

    print("\n=== MOST POSITIVE FEATURES ===")
    print(coefficients.sort_values(ascending=False).head(15))
    print("\n=== MOST NEGATIVE FEATURES ===")
    print(coefficients.sort_values(ascending=True).head(15))

    print("\n=== COMPARISON (higher score = spared) ===")
    for name_a, a, name_b, b in [
        ("Mother", ANNE, "5 strangers", BES_YABANCI),
        ("Baby", BEBEK, "Adult", YETISKIN),
    ]:
        sa, sb = puan(a, coeffs, X.columns), puan(b, coeffs, X.columns)
        print(f"{name_a}: {sa:.3f} | {name_b}: {sb:.3f} -> spare {name_a if sa > sb else name_b}")


if __name__ == '__main__':
    main()
