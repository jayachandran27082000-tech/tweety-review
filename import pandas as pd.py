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

# =====================================
# Load Train Dataset
# =====================================

df = pd.read_csv(base_dir / "train.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df["label"].value_counts())

# =====================================
# Text Cleaning Function
# =====================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return text

# =====================================
# Clean Train Tweets
# =====================================

df["clean_tweet"] = df["tweet"].apply(clean_text)

df["tokens"] = df["clean_tweet"].str.split()

print(df[["tweet","clean_tweet"]].head())

# =====================================
# Load Test Dataset
# =====================================

test_df = pd.read_csv(base_dir / "test.csv")

test_df["clean_tweet"] = test_df["tweet"].apply(clean_text)

test_df["tokens"] = test_df["clean_tweet"].str.split()

# =====================================
# Train Validation Split
# =====================================

train_texts, val_texts, train_labels, val_labels = train_test_split(

    df["clean_tweet"],

    df["label"],

    test_size=0.2,

    random_state=42,

    stratify=df["label"]

)

print("Training Samples :", len(train_texts))
print("Validation Samples :", len(val_texts))

# =====================================
# TF-IDF Vectorizer
# =====================================

tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(train_texts)

X_val = tfidf.transform(val_texts)

X_test = tfidf.transform(test_df["clean_tweet"])

# =====================================
# Convert to Tensor
# =====================================

X_train = torch.tensor(
    X_train.toarray(),
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val.toarray(),
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test.toarray(),
    dtype=torch.float32
)

y_train = torch.tensor(
    train_labels.values,
    dtype=torch.long
)

y_val = torch.tensor(
    val_labels.values,
    dtype=torch.long
)

# =====================================
# Dataset
# =====================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

# =====================================
# DataLoader
# =====================================

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False

)

# =====================================
# Neural Network
# =====================================

class HateSpeechModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.fc1 = nn.Linear(input_size,128)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(128,64)

        self.fc3 = nn.Linear(64,2)

    def forward(self,x):

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.fc3(x)

        return x

# =====================================
# Create Model
# =====================================

input_size = X_train.shape[1]

model = HateSpeechModel(input_size)

print(model)

# =====================================
# Loss Function
# =====================================

criterion = nn.CrossEntropyLoss()

# =====================================
# Optimizer
# =====================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.001

)

# =====================================
# Training
# =====================================

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss : {running_loss:.4f}")

# =====================================
# Validation
# =====================================

model.eval()

predictions = []

actual = []

with torch.no_grad():

    for features, labels in val_loader:

        outputs = model(features)

        _, predicted = torch.max(outputs,1)

        predictions.extend(predicted.numpy())

        actual.extend(labels.numpy())

accuracy = accuracy_score(actual,predictions)

print("\nValidation Accuracy :",accuracy)

# =====================================
# Predict Test Dataset
# =====================================

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    _, test_predictions = torch.max(outputs,1)

print("\nFirst 20 Predictions")

print(test_predictions[:20])

# =====================================
# Save Prediction File
# =====================================

submission = pd.DataFrame({

    "id":test_df["id"],

    "label":test_predictions.numpy()

})

submission.to_csv("submission.csv",index=False)

print("\nsubmission.csv saved successfully.")

sentence = input("Enter a tweet: ")

# Clean the sentence
clean_sentence = clean_text(sentence)

# Convert to TF-IDF
sentence_vector = tfidf.transform([clean_sentence])

# Convert to PyTorch tensor
sentence_tensor = torch.tensor(sentence_vector.toarray(), dtype=torch.float32)
# Make prediction
model.eval()   # Set model to evaluation mode

with torch.no_grad():
    output = model(sentence_tensor)
    prediction = torch.argmax(output, dim=1).item()

# Display result
if prediction == 0:
    print("Prediction: Non-Hate")
else:
    print("Prediction: Hate Speech")