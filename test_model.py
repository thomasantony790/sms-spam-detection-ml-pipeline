import joblib

# Load the trained model
model = joblib.load("/Users/thomasantony/Desktop/Spam SMS Project/spam_model.joblib")

print("SMS Spam Detector")
print("Type a message and press Enter.")
print("Type 'exit' to quit.\n")

while True:
    message = input("Message: ")

    if message.lower() == "exit":
        break

    # Predict
    prediction = model.predict([message])[0]

    label = "SPAM" if prediction == 1 else "HAM"

    # Optional confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([message])[0]
        confidence = max(proba)
        print(f"Prediction: {label} (confidence: {confidence:.3f})\n")
    else:
        print(f"Prediction: {label}\n")
