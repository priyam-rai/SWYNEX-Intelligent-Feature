"""
SWYNEX Internship - Task 3: Intelligent Feature
Customer Support Email Classifier (improved prototype)

What is new compared with Task 2:
1. Larger training set (80 emails) and a harder test set (24 emails).
2. Evaluation report: accuracy, per-category scores and a confusion table.
3. Failure cases: the emails the model gets wrong are printed and explained.
4. Intelligent feature: if the model is not confident, the email is marked
   "Needs human review" instead of guessing.

How to run:
    pip install -r requirements.txt
    python email_classifier.py                      (full evaluation report)
    python email_classifier.py "Your email text"    (classify one email)
"""

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

CATEGORIES = ["Billing", "Technical Issue", "Refund / Cancellation", "General Inquiry"]

# Below this confidence the email is sent to a human instead of being guessed
REVIEW_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# 1. Training data (written by hand for this demo, no personal data)
# ---------------------------------------------------------------------------
TRAIN_DATA = [
    # Billing
    ("I was charged twice for my last payment", "Billing"),
    ("Please send me the invoice for this month", "Billing"),
    ("My credit card payment failed but money was deducted", "Billing"),
    ("There is a wrong amount on my bill", "Billing"),
    ("Can you update the billing address on my invoice", "Billing"),
    ("I see an extra charge on my statement", "Billing"),
    ("How can I change my payment method", "Billing"),
    ("The tax amount on my invoice looks incorrect", "Billing"),
    ("My payment is showing as pending for three days", "Billing"),
    ("I did not receive a receipt after paying", "Billing"),
    ("Why was I billed more than the price shown", "Billing"),
    ("I need a copy of my receipt for last week's payment", "Billing"),
    ("My bank shows a charge that I do not recognise", "Billing"),
    ("The discount code was not applied to my final bill", "Billing"),
    ("Please update my card details for the next payment", "Billing"),
    ("I was charged after my free trial ended without notice", "Billing"),
    ("Can I get a GST invoice for my purchase", "Billing"),
    ("The amount deducted is different from the order total", "Billing"),
    ("My payment went through but the order shows unpaid", "Billing"),
    ("I paid online and the money is not credited yet", "Billing"),
    # Technical Issue
    ("The app crashes every time I open it", "Technical Issue"),
    ("I cannot log in to my account", "Technical Issue"),
    ("The website is not loading on my phone", "Technical Issue"),
    ("I get an error message when I upload a file", "Technical Issue"),
    ("The password reset email never arrives", "Technical Issue"),
    ("The page freezes when I click the submit button", "Technical Issue"),
    ("Notifications have stopped working after the update", "Technical Issue"),
    ("The video keeps buffering and then stops", "Technical Issue"),
    ("I found a bug in the search feature", "Technical Issue"),
    ("The app is very slow and keeps hanging", "Technical Issue"),
    ("I keep getting a blank screen after the latest update", "Technical Issue"),
    ("The login page says invalid password even though it is correct", "Technical Issue"),
    ("I am unable to download my report", "Technical Issue"),
    ("The mobile app logs me out every few minutes", "Technical Issue"),
    ("Images are not showing on the product page", "Technical Issue"),
    ("I did not get the verification code on my phone", "Technical Issue"),
    ("The checkout button does not respond", "Technical Issue"),
    ("Error 500 appears when I open the dashboard", "Technical Issue"),
    ("The app does not work on my new phone", "Technical Issue"),
    ("Sound is missing in the video lessons", "Technical Issue"),
    # Refund / Cancellation
    ("I want a refund for my last order", "Refund / Cancellation"),
    ("Please cancel my subscription", "Refund / Cancellation"),
    ("I would like my money back for this product", "Refund / Cancellation"),
    ("How do I cancel my order before it ships", "Refund / Cancellation"),
    ("The item arrived damaged and I want a refund", "Refund / Cancellation"),
    ("Please stop my monthly plan and refund this month", "Refund / Cancellation"),
    ("I ordered by mistake, please cancel it", "Refund / Cancellation"),
    ("I want to return this product and get a refund", "Refund / Cancellation"),
    ("Cancel my account and refund the remaining balance", "Refund / Cancellation"),
    ("I am not satisfied and need my money returned", "Refund / Cancellation"),
    ("I was not able to use the service, please give me my money back", "Refund / Cancellation"),
    ("Please cancel my plan before it renews", "Refund / Cancellation"),
    ("I want to return the shoes I bought last week", "Refund / Cancellation"),
    ("Wrong item was delivered, I need a replacement or refund", "Refund / Cancellation"),
    ("Stop my auto renewal and cancel the subscription", "Refund / Cancellation"),
    ("Please process a refund to my original payment method", "Refund / Cancellation"),
    ("I changed my mind and want to cancel the purchase", "Refund / Cancellation"),
    ("How long will my refund take to arrive", "Refund / Cancellation"),
    ("I want to close my account and get a refund", "Refund / Cancellation"),
    ("Please cancel my booking and return the amount", "Refund / Cancellation"),
    # General Inquiry
    ("What are your customer support working hours", "General Inquiry"),
    ("Do you ship to other countries", "General Inquiry"),
    ("Can you tell me more about your premium plan features", "General Inquiry"),
    ("Is there a student discount available", "General Inquiry"),
    ("I just wanted to say your service is great", "General Inquiry"),
    ("Where can I find your product catalogue", "General Inquiry"),
    ("Do you offer a free trial", "General Inquiry"),
    ("How do I contact your sales team", "General Inquiry"),
    ("Which languages does your product support", "General Inquiry"),
    ("I have a suggestion for a new feature", "General Inquiry"),
    ("Do you have an offline version of the app", "General Inquiry"),
    ("What is the difference between the basic and pro plan", "General Inquiry"),
    ("Can I use one account on two devices", "General Inquiry"),
    ("Are there any upcoming sales or offers", "General Inquiry"),
    ("Do you provide certificates after the course", "General Inquiry"),
    ("Where is your company located", "General Inquiry"),
    ("Is your product available in Hindi", "General Inquiry"),
    ("How can I become a reseller partner", "General Inquiry"),
    ("Thank you for the quick help yesterday", "General Inquiry"),
    ("Do you have a mobile app for Android", "General Inquiry"),
]

# ---------------------------------------------------------------------------
# 2. Test data (never used for training). The last two emails in each group
#    are deliberately harder: short, unusual wording or two issues at once.
# ---------------------------------------------------------------------------
TEST_DATA = [
    # Billing
    ("I was billed two times for the same order", "Billing"),
    ("Please share the invoice for last month's payment", "Billing"),
    ("There is a wrong charge on my card statement", "Billing"),
    ("The price on my bill is higher than what was advertised", "Billing"),
    ("Money was taken from my account but I got no confirmation", "Billing"),
    ("Please send a receipt for my last transaction", "Billing"),
    # Technical Issue
    ("The app keeps crashing when I try to log in", "Technical Issue"),
    ("I get an error when I try to upload my photo", "Technical Issue"),
    ("The website is very slow and not loading", "Technical Issue"),
    ("I never received the OTP to verify my account", "Technical Issue"),
    ("The page goes blank after I press pay", "Technical Issue"),
    ("Videos will not play on my laptop", "Technical Issue"),
    # Refund / Cancellation
    ("I want my money back for this order", "Refund / Cancellation"),
    ("Please cancel my subscription from next month", "Refund / Cancellation"),
    ("I want to return the damaged item and get a refund", "Refund / Cancellation"),
    ("Cancel my order, I ordered the wrong size", "Refund / Cancellation"),
    ("I was charged twice, please refund the extra payment", "Refund / Cancellation"),
    ("I am not happy with the course and want my fees back", "Refund / Cancellation"),
    # General Inquiry
    ("What time does your support team work", "General Inquiry"),
    ("Do you have any discount for students", "General Inquiry"),
    ("Can you tell me about the features of your plans", "General Inquiry"),
    ("Is there an app for iPhone", "General Inquiry"),
    ("I love the new design, great job", "General Inquiry"),
    ("Does the premium plan include a certificate", "General Inquiry"),
]

# Extra emails used only for the demo (not part of the evaluation score)
DEMO_EMAILS = [
    "Hi, I was charged twice this month. Please check my payment.",
    "The app crashes whenever I try to open my profile.",
    "I want to cancel my order and get a refund.",
    "Do you offer any discount for new customers?",
    "Hello",
    "I was charged twice and the app also crashed, please help",
]


# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
def build_model():
    """TF-IDF turns text into numbers, Naive Bayes picks the category."""
    texts = [text for text, _ in TRAIN_DATA]
    labels = [label for _, label in TRAIN_DATA]
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
        MultinomialNB(alpha=0.1),
    )
    model.fit(texts, labels)
    return model


def classify(model, email_text):
    """Return (category, confidence, needs_review) for one email."""
    probabilities = model.predict_proba([email_text])[0]
    best_index = probabilities.argmax()
    category = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    needs_review = confidence < REVIEW_THRESHOLD
    return category, confidence, needs_review


# ---------------------------------------------------------------------------
# 4. Evaluation report
# ---------------------------------------------------------------------------
def evaluate(model):
    texts = [text for text, _ in TEST_DATA]
    true_labels = [label for _, label in TEST_DATA]
    predictions = list(model.predict(texts))

    accuracy = accuracy_score(true_labels, predictions)
    print("=" * 60)
    print(f"OVERALL ACCURACY on {len(TEST_DATA)} unseen emails: {accuracy:.0%}")
    print("=" * 60)

    print("\nPer-category scores")
    print(classification_report(true_labels, predictions, labels=CATEGORIES, zero_division=0))

    print("Confusion table (rows = true category, columns = predicted)")
    matrix = confusion_matrix(true_labels, predictions, labels=CATEGORIES)
    short = ["Bill", "Tech", "Refund", "General"]
    print(f"{'':24}" + "".join(f"{name:>9}" for name in short))
    for category, row in zip(CATEGORIES, matrix):
        print(f"{category:24}" + "".join(f"{count:>9}" for count in row))

    print("\nFailure cases (emails the model got wrong)")
    failures = 0
    for text, true, predicted in zip(texts, true_labels, predictions):
        if true != predicted:
            failures += 1
            _, confidence, _ = classify(model, text)
            print(f'- "{text}"')
            print(f"    true: {true} | predicted: {predicted} ({confidence:.0%})")
    if failures == 0:
        print("- none")

    print(f"\nHuman-review feature (threshold = {REVIEW_THRESHOLD:.0%})")
    accepted_correct = accepted = flagged = flagged_wrong = 0
    for text, true in TEST_DATA:
        category, _, needs_review = classify(model, text)
        if needs_review:
            flagged += 1
            if category != true:
                flagged_wrong += 1
        else:
            accepted += 1
            if category == true:
                accepted_correct += 1
    if accepted:
        print(f"- Answered automatically: {accepted} emails, {accepted_correct} correct "
              f"({accepted_correct / accepted:.0%})")
    print(f"- Sent to a human: {flagged} emails, of which {flagged_wrong} would have been wrong")


def demo(model):
    print("\nDemo on new emails")
    print("-" * 60)
    for email in DEMO_EMAILS:
        category, confidence, needs_review = classify(model, email)
        result = "Needs human review" if needs_review else category
        print(f"Input  : {email}")
        print(f"Output : {result} (model's best guess: {category}, {confidence:.0%})\n")


def main():
    model = build_model()

    if len(sys.argv) > 1:
        email = " ".join(sys.argv[1:])
        category, confidence, needs_review = classify(model, email)
        result = "Needs human review" if needs_review else category
        print(f"Email  : {email}")
        print(f"Output : {result} (model's best guess: {category}, {confidence:.0%})")
        return

    evaluate(model)
    demo(model)


if __name__ == "__main__":
    main()
