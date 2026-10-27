# Ethical Decision-Making for Autonomous Vehicles

A data-driven decision model for unavoidable-collision dilemmas, learned from a survey with **12,414 participants**. Each participant faced 25 trolley-style scenarios: a car with failed brakes must hit one of two groups. The project turns the answers into a scoring function that compares the two groups.

Authors: Mert Gürlek, Zeynep Görü, Rüveyda Tüfekçioğlu

## Survey

**Form.** "Etik Farkındalık Formu" (Ethics Awareness Form), a Google Form in Turkish. The dataset and the questionnaire file are not included in this repository.

- **Intro text.** The participant is told to imagine driving a car with failed brakes. The car cannot stop and will certainly hit someone. Steering away sends it onto another road, where other people or animals will also certainly be hurt. "Your decision determines which side is harmed." The form states that it is a voluntary thought experiment with no right or wrong answers, that no personal data is requested or recorded, and that it is based on the trolley problem (Foot, 1967).
- **Demographics.** Age and gender are required (Male / Female).
- **Questions.** 25 required single-choice questions, each titled "Which road do you steer onto?" with two options.
- **Meaning of an answer.** The chosen option is the group that gets hit. The other group is spared.
- **Design.** Each question changes one factor at a time, so every answer isolates one preference: species, health, age, fame, closeness (own mother vs. a stranger), moral character, wealth, gender, nationality, body weight, pregnancy, pedestrian vs. cyclist, and group size. Several factors are asked twice, once with equal group sizes and once with the numbers against the preferred group (for example Q9 and Q10, or Q17, Q18 and Q19).
- **Sample.** 12,414 responses collected from August 2025 to June 2026. 8,568 respondents (69%) are female and 3,846 (31%) are male. The median age is 21, and about 70% of the respondents are between 18 and 24.

### Share of respondents who spared each group

| Q | Option A | Option B | Spared A | Spared B |
|---|---|---|---|---|
| 1 | 1 cat | 1 dog | 66.2% | 33.8% |
| 2 | 1 disabled person | 1 healthy person | 20.9% | 79.1% |
| 3 | 1 person | 3 people | 3.8% | 96.2% |
| 4 | 1 baby | 1 adult | 76.6% | 23.4% |
| 5 | 1 person | 2 cats | 72.9% | 27.1% |
| 6 | 1 healthy person | 3 disabled people | 41.5% | 58.5% |
| 7 | 1 celebrity | 1 unknown person | 42.3% | 57.7% |
| 8 | 1 celebrity | 3 unknown people | 6.8% | 93.2% |
| 9 | your mother | 1 stranger | 98.0% | 2.0% |
| 10 | your mother | 5 strangers | 90.7% | 9.3% |
| 11 | 1 bad person | 1 good person | 1.1% | 98.9% |
| 12 | 3 bad people | 1 good person | 6.2% | 93.8% |
| 13 | 1 man | 1 woman | 6.7% | 93.3% |
| 14 | 1 poor person | 1 rich person | 68.8% | 31.2% |
| 15 | 1 rich person | 3 poor people | 13.3% | 86.7% |
| 16 | 1 dog | 3 cats | 9.6% | 90.4% |
| 17 | 1 Turkish | 1 foreign national | 83.5% | 16.5% |
| 18 | 1 foreign national | 2 Turkish | 8.0% | 92.0% |
| 19 | 1 Turkish | 2 foreign nationals | 51.9% | 48.1% |
| 20 | 1 thin person | 2 overweight people | 30.2% | 69.8% |
| 21 | 1 thin man | 1 overweight man | 77.9% | 22.1% |
| 22 | 1 adult | 2 babies | 11.6% | 88.4% |
| 23 | pregnant woman | adult | 97.9% | 2.1% |
| 24 | cyclist | pedestrian | 32.0% | 68.0% |
| 25 | 1 pregnant woman | 2 adults | 73.8% | 26.2% |

Two examples: 98.9% spare the good person over the bad one (Q11), and 97.9% spare a pregnant woman over an adult (Q23).

## Algorithms and methods

### 1. Preprocessing

- The Google Forms export is wide, with one row per respondent and one column per question.
- Each answer text (for example `3 ENGELLİ İNSAN`) is parsed with keyword rules into the group size (numeric) and 12 categorical attributes: `varlık` (species), `sağlık` (health), `yaş` (age group), `ünlü` (fame), `bağlılık` (closeness), `iyilik` (moral character), `zenginlik` (wealth), `cinsiyet` (gender), `uyruk` (nationality), `kilo` (body weight), `hamile` (pregnancy), `araç` (pedestrian or cyclist).
- An attribute the question does not mention is set to `belirtilmemiş` (not specified), so nothing is assumed about the group.
- Typos in the responses are merged (`ANNEN` and `ANNENİZ`).
- The data is reshaped to the long format: for every respondent and question there are two rows, one per group. The spared group gets `tercih_edildi = 1` and the hit group gets 0. This gives 12,414 × 25 × 2 = **620,700 rows**.

### 2. Model (`ethical_decision_model.py`)

- **Encoding.** The 12 categorical columns are one-hot encoded with pandas `get_dummies`. The group size stays numeric. The result is 35 features.
- **Algorithm.** Logistic regression (scikit-learn, `lbfgs` solver, L2 regularization with C = 1.0, `max_iter = 1000`) predicts the probability that a group is spared.
- **Why logistic regression.** The coefficients can be read directly: a positive weight means a group with that attribute is more likely to be spared, and a negative weight means it is more likely to be hit. A black-box model would not show which factors drive the decisions.

### 3. Decision rule

The score of a group is

```
score = w_size × group size + Σ w_attribute   (over the group's attributes)
```

and the vehicle spares the group with the higher score. `puan()` implements this and warns if an attribute value was never seen in the survey.

### 4. Evaluation

- **Split.** 80/20 by respondent (`random_state = 42`), so nobody appears in both the training and test sets. A row-level split would leak, because the two rows of a question come from the same person.
- **Metrics.** Single-group accuracy (does the model predict spared or hit for one group) and pairwise accuracy (does the spared group get the higher score of the two). The pairwise one matches how the score is used.

## Results

| Metric (held-out respondents) | Value |
|---|---|
| Accuracy, single group (spared or hit) | 75.6% |
| Accuracy, which of the two groups is spared | **79.2%** |

The model is trained on 620,700 rows and converges in 46 iterations.

### Learned weights

| Attribute value | Weight | Attribute value | Weight |
|---|---|---|---|
| closeness: own mother | +3.80 | closeness: stranger | -5.13 |
| moral character: good | +3.65 | moral character: bad | -4.35 |
| gender: woman | +1.92 | age: adult | -1.82 |
| age: baby | +1.31 | gender: man | -1.52 |
| group size (per person) | +1.01 | nationality: foreign | -1.43 |
| nationality: Turkish | +0.98 | health: disabled | -1.28 |
| health: healthy | +0.68 | wealth: rich | -1.12 |
| wealth: poor | +0.66 | fame: celebrity | -0.99 |
| fame: not famous | +0.51 | species: dog | -0.81 |
| body weight: thin | +0.32 | pregnant | -0.73 |
| species: human | +0.22 | cyclist | -0.55 |
| not pregnant | +0.14 | body weight: overweight | -0.54 |

Weights for `belirtilmemiş` (not specified) are left out of the table. They are in the full output of `ethical_decision_model.py`.

### Findings

- **Closeness and moral character dominate.** 98% spare their own mother over a stranger (Q9), and 99% spare a good person over a bad one (Q11). Even against 5 strangers, 91% spare their mother (Q10). Even when three bad people face one good person, 94% spare the good one (Q12).
- **Numbers matter a lot, but not always.** 96% choose 3 people over 1 (Q3) and 93% choose 3 unknown people over 1 celebrity (Q8). People split on 1 healthy person vs. 3 disabled people (41.5% spare the healthy one, Q6).
- **Vulnerable groups are protected.** Babies over adults (77% in Q4, 88% in Q22 even when the baby group is 2 against 1 adult), and a pregnant woman over an adult (98%, Q23).
- **Humans over animals, but with limits.** 73% spare a person over 2 cats (Q5), while 90% spare 3 cats over 1 dog (Q16), so size outweighs species in the animal pairs.
- **Group preferences.** Women over men (93%, Q13), Turkish over foreign nationals (83.5% with equal size in Q17, but 52% when the foreign group has 2 people in Q19), thin over overweight (78%, Q21). 69% spare the poor person over the rich one in a 1-vs-1 case (Q14), and 87% spare 3 poor people over 1 rich person (Q15).
- **Pedestrians over cyclists** are spared by 68% (Q24).

### How to read the weights

- The weights are relative to the other values of the same attribute, and are fitted jointly. A single weight is not an independent effect. For example, "pregnant" has a negative weight even though 98% spare the pregnant woman in Q23. That group is also a woman (+1.92), and the comparison group is an adult (-1.82). Always compare **total scores** of two groups.
- Examples from the scoring function: a mother beats 5 strangers (3.41 vs. -1.49) and a baby beats an adult (1.73 vs. -1.40), as most respondents chose.

## Usage

The survey data is not published with the code. To rerun the analysis, put the preprocessed table described in the Preprocessing section at `data/survey_long.csv`, and the raw Google Forms export at `data/form_responses.csv` for the summary table.

```bash
pip install pandas scikit-learn matplotlib
python ethical_decision_model.py    # train, evaluate, print weights and example decisions
python plot_results.py              # per-question table and docs/coefficients.png
```

`survey_long.csv` has these columns: `katılımcı` (respondent), `soru` (question), `kişi_sayısı` (group size), the 12 attribute columns, and `tercih_edildi` (1 = spared, 0 = hit).

## Repository layout

| Path | Description |
|---|---|
| `ethical_decision_model.py` | Training, evaluation and scoring |
| `plot_results.py` | Summary table and coefficient plot |

## Limitations and ethics

- The model describes the **preferences of the survey sample**, which is young (median age 21) and mostly female. It does not say what a vehicle *should* do.
- Several attributes, such as wealth, nationality, fame, gender and weight, would be discriminatory and illegal to use in a real vehicle. In this project they serve to analyze which factors drive public opinion. They are not a proposal for deployment.
- Each question changes one factor, so weights for attributes that never occur together are not tested. Scoring a made-up group that mixes many attributes extrapolates beyond the survey.
- The age field is self-reported and contains some invalid entries. The model does not use it.

Stack: Python, pandas, scikit-learn, matplotlib.
