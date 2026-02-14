import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_PATH = "/Users/thomasantony/Desktop/Spam SMS Project/spam.csv"
MODEL_PATH = "/Users/thomasantony/Desktop/Spam SMS Project/spam_model.joblib"

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
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LinearSVC(
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Report:")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))

    joblib.dump(model, MODEL_PATH)
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
