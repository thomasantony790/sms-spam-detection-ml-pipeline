import csv
import os
from datetime import datetime

USER_DATA_PATH = "/Users/thomasantony/Desktop/Spam SMS Project/user_messages.csv"

def ensure_file_exists():
    if not os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "text", "source", "created_at"])

def append_message(label: str, text: str, source: str = "web"):
    ensure_file_exists()

    clean_label = label.strip().lower()
    clean_text = text.strip()

    if clean_label not in ["ham", "spam"]:
        raise ValueError("label must be 'ham' or 'spam'")

    if not clean_text:
        raise ValueError("text is empty")

    with open(USER_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([clean_label, clean_text, source, datetime.utcnow().isoformat()])
