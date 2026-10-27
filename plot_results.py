"""Summarize the survey and plot the fitted weights (writes PNGs to docs/)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from ethical_decision_model import DEFAULT_DATA, ID_COL, fit_model, load_data

df = pd.read_csv(DEFAULT_DATA, encoding='utf-8-sig')

# Share of respondents who spared each group, per question (markdown table)
raw = pd.read_csv('data/form_responses.csv')
print('| Q | Option A | Option B | Spared A | Spared B |')
print('|---|---|---|---|---|')
for i, col in enumerate(raw.columns[3:], start=1):
    counts = raw[col].replace({'ANNEN': 'ANNENİZ'}).value_counts()
    (a, na), (b, nb) = sorted(counts.items())
    # the chosen option is the one hit, so the other option is spared
    print(f'| {i} | {a} | {b} | {nb / len(raw):.1%} | {na / len(raw):.1%} |')

X, y, _, _ = load_data(DEFAULT_DATA)
coefs = pd.Series(fit_model(X, y).coef_[0], index=X.columns)
coefs = coefs[~coefs.index.str.endswith('belirtilmemiş')].sort_values()

fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(coefs.index, coefs.values, color=['#c0504d' if v < 0 else '#4f81bd' for v in coefs.values])
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Logistic regression coefficient (higher = more likely to be spared)')
ax.set_title('Learned weight of each attribute')
fig.tight_layout()
fig.savefig('docs/coefficients.png', dpi=150)
