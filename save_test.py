from data_store import append_message

append_message("spam", "Test spam message with a fake link bit.ly/test", source="manual")
append_message("ham", "Hey are we still meeting at 7 tonight", source="manual")

print("Saved 2 rows to user_messages.csv")
