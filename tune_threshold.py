import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

DATA_PATH = "/Users/thomasantony/Desktop/Spam SMS Project/spam.csv"

def main():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str)
    df["y"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for t in thresholds:
        pred = (probs >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_test, pred)
        print("\nThreshold:", t)
        print("Spam precision:", round(p, 4), "Spam recall:", round(r, 4), "Spam f1:", round(f, 4))
        print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()
