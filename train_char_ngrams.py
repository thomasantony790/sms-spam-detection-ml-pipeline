import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

BASE_DATA_PATH = "/Users/thomasantony/Desktop/Spam SMS Project/spam.csv"
USER_DATA_PATH = "/Users/thomasantony/Desktop/Spam SMS Project/user_messages.csv"
MODEL_PATH = "models/spam_model_char.joblib"

def load_base_dataset():
    df = pd.read_csv(BASE_DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    return df

def load_user_dataset():
    if not os.path.exists(USER_DATA_PATH):
        return pd.DataFrame(columns=["label", "text"])

    if os.path.getsize(USER_DATA_PATH) == 0:
        return pd.DataFrame(columns=["label", "text"])

    df = pd.read_csv(USER_DATA_PATH, encoding="utf-8")

    if "label" not in df.columns or "text" not in df.columns:
        return pd.DataFrame(columns=["label", "text"])

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"].str.len() > 0]
    df = df[["label", "text"]]

    return df

def main():
    base_df = load_base_dataset()
    user_df = load_user_dataset()

    full_df = pd.concat([base_df, user_df], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["label", "text"])

    print("Base rows:", len(base_df))
    print("User rows:", len(user_df))
    print("Total rows used:", len(full_df))

    full_df["y"] = full_df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        full_df["text"],
        full_df["y"],
        test_size=0.2,
        random_state=42,
        stratify=full_df["y"]
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            )
        )
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=4))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nSaved:", MODEL_PATH)

if __name__ == "__main__":
    main()
