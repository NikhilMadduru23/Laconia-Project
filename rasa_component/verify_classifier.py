from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "intent_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Test input
text = "I want a flight from Charlotte to Las Vegas"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Assuming you have a label map from training (e.g., 0: "atis_flight", 1: "atis_airfare", etc.)
label_map = {0: "atis_abbreviation", 1: "atis_aircraft", 2: "atis_airfare", 3: "atis_airline", 
             4: "atis_flight", 5: "atis_flight_time", 6: "atis_ground_service", 7: "atis_quantity"}  # Adjust this!
predicted_intent = label_map[predicted_class]
print(f"Predicted intent: {predicted_intent}")