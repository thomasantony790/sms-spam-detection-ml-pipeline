from flask import Flask, render_template, request, url_for
import joblib
from data_store import append_message

app = Flask(__name__)
model = joblib.load("models/spam_model_char.joblib")


def predict_label_and_confidence(text: str):
    pred = model.predict([text])[0]
    label = "spam" if int(pred) == 1 else "ham"

    confidence = None
    try:
        proba = model.predict_proba([text])[0]
        confidence = float(max(proba))
    except Exception:
        confidence = None

    return label, confidence


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        message="",
        label=None,
        confidence=None,
        error=None,
        success=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template(
            "index.html",
            message="",
            label=None,
            confidence=None,
            error="Please paste a message first.",
            success=None,
        )

    label, confidence = predict_label_and_confidence(message)

    return render_template(
        "index.html",
        message=message,
        label=label,
        confidence=confidence,
        error=None,
        success=None,
    )


@app.route("/save", methods=["POST"])
def save():
    message = (request.form.get("message") or "").strip()
    confirmed = (request.form.get("confirmed") or "").strip().lower()

    if not message:
        return render_template(
            "index.html",
            message="",
            label=None,
            confidence=None,
            error="Message missing.",
            success=None,
        )

    if confirmed not in ["ham", "spam"]:
        label, confidence = predict_label_and_confidence(message)
        return render_template(
            "index.html",
            message=message,
            label=label,
            confidence=confidence,
            error="Pick ham or spam before saving.",
            success=None,
        )

    append_message(confirmed, message, source="web")

    label, confidence = predict_label_and_confidence(message)
    return render_template(
        "index.html",
        message=message,
        label=label,
        confidence=confidence,
        error=None,
        success="Saved. Thank you.",
    )


@app.route("/about", methods=["GET"])
def about():
    return render_template(
        "about.html",
        model_name="TF-IDF + Logistic Regression",
        vectorizer_type="Character n-grams TF-IDF",
        dataset_name="SMS Spam Collection (UCI)",
        dataset_size="5,574 SMS messages",
        dataset_notes="English SMS labeled ham or spam. Donated in 2012.",
        save_notes="Use the Save control after a prediction to log the message with the correct label for future retraining.",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
