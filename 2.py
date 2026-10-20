from pathlib import Path

import pandas as pd
import re
import string
import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

base_dir = Path(__file__).resolve().parent

# ==========================
# Load Dataset
# ==========================

train_df = pd.read_csv(base_dir / "train.csv")
test_df = pd.read_csv(base_dir / "test.csv")

# ==========================
# Text Cleaning
# ==========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

train_df["clean_tweet"] = train_df["tweet"].apply(clean_text)
test_df["clean_tweet"] = test_df["tweet"].apply(clean_text)

# ==========================
# Split Data
# ==========================

X_train, X_val, y_train, y_val = train_test_split(
    train_df["clean_tweet"],
    train_df["label"],
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"]
)

# ==========================
# TF-IDF
# ==========================

tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(X_train)
X_val = tfidf.transform(X_val)
X_test = tfidf.transform(test_df["clean_tweet"])

# ==========================
# Convert to Tensor
# ==========================

X_train = torch.FloatTensor(X_train.toarray())
X_val = torch.FloatTensor(X_val.toarray())
X_test = torch.FloatTensor(X_test.toarray())

y_train = torch.LongTensor(y_train.values)
y_val = torch.LongTensor(y_val.values)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=32
)

# ==========================
# Model
# ==========================

class HateSpeechModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        return self.network(x)

model = HateSpeechModel(X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================
# Training
# ==========================

epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(output, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# ==========================
# Validation
# ==========================

model.eval()

predictions = []

actual = []

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        output = model(X_batch)

        pred = torch.argmax(output, dim=1)

        predictions.extend(pred.numpy())

        actual.extend(y_batch.numpy())

accuracy = accuracy_score(actual, predictions)

print("Validation Accuracy:", accuracy)

# ==========================
# Test Prediction
# ==========================

with torch.no_grad():

    test_output = model(X_test)

    test_prediction = torch.argmax(test_output, dim=1)

submission = pd.DataFrame({
    "id": test_df["id"],
    "label": test_prediction.numpy()
})

submission.to_csv("submission.csv", index=False)

print("submission.csv saved.")

# ==========================
# User Prediction
# ==========================

sentence = input("Enter a Tweet: ")

sentence = clean_text(sentence)

vector = tfidf.transform([sentence])

tensor = torch.FloatTensor(vector.toarray())

with torch.no_grad():

    prediction = torch.argmax(model(tensor), dim=1).item()

print("Prediction:", "Non-Hate" if prediction == 0 else "Hate Speech")