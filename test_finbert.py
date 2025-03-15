import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Load FinBERT model & tokenizer
model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Sample news article
sample_text = "Stock market surges after strong earnings reports."

# Tokenize input
encoding = tokenizer(sample_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

# Predict sentiment
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

# Sentiment mapping
sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
predicted_sentiment = sentiment_map.get(predicted_class, "UNKNOWN")

# Print result
print(f" News: {sample_text}")
print(f"Predicted Sentiment: {predicted_sentiment}")
