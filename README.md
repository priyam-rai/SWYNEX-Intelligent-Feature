# SWYNEX Intelligent Feature: Customer Support Email Classifier

This is the Task 3 submission for the SWYNEX AI Internship. It improves the Task 2 prototype, which sorts customer support emails into Billing, Technical Issue, Refund / Cancellation, or General Inquiry.

## What is new compared with Task 2

1. **More data and a harder test:** 80 training emails (was 40) and 24 unseen test emails (was 12). Some test emails are deliberately tricky (short, unusual wording, or two issues in one email).
2. **Evaluation report:** accuracy, per-category scores and a confusion table.
3. **Failure cases:** the wrong predictions are listed and explained below.
4. **Intelligent feature (human review):** if the model is not confident, the email is marked "Needs human review" instead of being guessed.
5. **Simple web interface:** paste an email in a browser page and see the result.

## Files

- `email_classifier.py`: data, model, evaluation report and human-review feature
- `app.py`: simple web interface (Python standard library only)
- `requirements.txt`: the library needed (scikit-learn)
- `README.md`: this file

## How to run

```
pip install -r requirements.txt
python email_classifier.py
python app.py
```

`email_classifier.py` prints the full evaluation report and a demo. `app.py` starts the web interface at http://localhost:8000.

To classify a single email from the command line:

```
python email_classifier.py "I cannot reset my password"
```

No API keys are used or stored.

## Evaluation results

Model: TF-IDF + Multinomial Naive Bayes (scikit-learn). Test set: 24 emails the model never saw in training (6 per category).

**Overall accuracy: 92% (22 of 24 correct).**

| Category | Precision | Recall |
|---|---|---|
| Billing | 0.75 | 1.00 |
| Technical Issue | 1.00 | 0.83 |
| Refund / Cancellation | 1.00 | 0.83 |
| General Inquiry | 1.00 | 1.00 |

Confusion table (rows = true category, columns = predicted):

| True \ Predicted | Billing | Technical | Refund | General |
|---|---|---|---|---|
| Billing | 6 | 0 | 0 | 0 |
| Technical Issue | 1 | 5 | 0 | 0 |
| Refund / Cancellation | 1 | 0 | 5 | 0 |
| General Inquiry | 0 | 0 | 0 | 6 |

The accuracy is lower than the 100% reported in Task 2 because the test set is larger and harder, so this number is a more honest picture of the model.

## Failure cases

| Email | True category | Predicted | Confidence |
|---|---|---|---|
| "Videos will not play on my laptop" | Technical Issue | Billing | 47% |
| "I was charged twice, please refund the extra payment" | Refund / Cancellation | Billing | 85% |

- **Case 1:** the email uses words that do not appear in the training emails (for example "play" and "laptop"), so the model had little to go on and was unsure. The human-review feature catches this one.
- **Case 2:** the email contains two ideas, a billing problem ("charged twice") and a refund request. The model picked Billing with high confidence. This is the most important weakness, because the confidence score cannot warn us about it.

## Intelligent feature: human review

The model gives a confidence score for every prediction. If the confidence is below **50%**, the output is **"Needs human review"** instead of a category. This matches the Task 1 constraint that a human always has the final say.

On the test set:
- 22 emails were answered automatically and 21 were correct (95%).
- 2 emails were sent to a human, and 1 of those 2 would have been wrong.

Note: the 50% threshold was chosen after looking at the confidence values on this small test set, so these numbers are likely optimistic. A fair check needs a separate validation set.

Example outputs:

| Input | Output |
|---|---|
| Hi, I was charged twice this month. Please check my payment. | Billing (94%) |
| The app crashes whenever I try to open my profile. | Technical Issue (91%) |
| I want to cancel my order and get a refund. | Refund / Cancellation (100%) |
| Hello | Needs human review (best guess Billing, 25%) |

## Web interface

`app.py` starts a small web page. Paste an email, press **Classify**, and it shows the category and confidence, or a yellow "Needs human review" box when the model is unsure.

## Limitations

- The dataset is small and written by hand, so results on real customer emails may be different.
- Emails with two issues in one can be confidently misclassified (failure case 2).
- Only English is supported.
- The review threshold should be tuned on separate data.

## Next steps

- Collect more real emails and add a separate validation set.
- Allow more than one category per email.
- Compare with a language model or an AI API on the same test set.
