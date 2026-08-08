import re
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


THRESHOLD = 0.60


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if len(word) > 1]

    return " ".join(words)


def priority_tag(text):
    text = text.lower()

    urgent_words = [
        "urgent", "critical", "down", "outage",
        "not working", "failed", "failure",
        "broken", "crash", "immediately", "asap"
    ]

    for word in urgent_words:
        if word in text:
            return "Urgent"

    return "Normal"


def load_data():
    df = pd.read_csv("tickets.csv")

    df["text"] = (
        df["subject"].fillna("") + " " +
        df["body"].fillna("")
    )

    df["clean_text"] = df["text"].apply(clean_text)

    return df


def train_model(df):
    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nMODEL EVALUATION")
    print("-" * 40)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", f"{accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    return vectorizer, model


def classify_ticket(text, vectorizer, model):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(features)[0]
    index = np.argmax(probabilities)

    predicted_category = model.classes_[index]
    confidence = probabilities[index]

    review = confidence < THRESHOLD
    priority = priority_tag(text)

    if review:
        category = "Needs Human Review"
    else:
        category = predicted_category

    return category, predicted_category, confidence, priority, review


def test_new_tickets(vectorizer, model):
    tickets = [
        "My card was charged twice for the same purchase. Please refund the extra amount.",
        "The application keeps crashing whenever I open the dashboard.",
        "I want to know how I can apply for parental leave.",
        "Can you provide more information about your available plans?",
        "URGENT: The production system is down and customers cannot complete their orders."
    ]

    print("\nPREDICTIONS ON 5 NEW TICKETS")
    print("-" * 40)

    for i, ticket in enumerate(tickets, 1):
        category, raw_category, confidence, priority, review = classify_ticket(
            ticket, vectorizer, model
        )

        print(f"\nTicket {i}: {ticket}")
        print("Category:", category)
        print("Confidence:", f"{confidence:.1%}")
        print("Priority:", priority)
        print("Human review:", "Yes" if review else "No")


def run_cli(vectorizer, model):
    print("\nLIVE TICKET CLASSIFIER")
    print("Type a ticket and press Enter.")
    print("Type 'exit' to stop.")

    while True:
        text = input("\nNew ticket: ").strip()

        if text.lower() == "exit":
            print("Demo ended.")
            break

        if not text:
            print("Please enter some ticket text.")
            continue

        category, raw_category, confidence, priority, review = classify_ticket(
            text, vectorizer, model
        )

        print("Category:", category)
        print("Confidence:", f"{confidence:.1%}")
        print("Priority:", priority)

        if review:
            print("Action: Send to human review")
        else:
            print("Action: Auto assign")


def main():
    df = load_data()

    print("Support Ticket Classification")
    print("Number of tickets:", len(df))
    print("\nCategories:")
    print(df["category"].value_counts())

    vectorizer, model = train_model(df)

    # Required: test the model on five new tickets
    test_new_tickets(vectorizer, model)

    # Bonus: interactive CLI demo
    if "--demo" in sys.argv:
        run_cli(vectorizer, model)


if __name__ == "__main__":
    main()