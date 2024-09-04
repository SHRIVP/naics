# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import NAICSClassifier
import pandas as pd
import tiktoken
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
file_path = './data/naics_classifier_examples_detailed.csv'
df = pd.read_csv(file_path)

# Initialize the tokenizer using 'cl100k_base' encoding
tokenizer = tiktoken.get_encoding('cl100k_base')
vocab_size = tokenizer.n_vocab

# Tokenize the 'Business Description' column
df['Tokenized Description'] = df['Business Description'].apply(lambda x: tokenizer.encode(x))

# Encode the NAICS codes as integers
label_encoder = LabelEncoder()
df['Encoded NAICS Code'] = label_encoder.fit_transform(df['NAICS Code'])

# Convert the tokenized descriptions to tensors
tokenized_sequences = df['Tokenized Description'].tolist()
tokenized_sequences = [torch.tensor(seq) for seq in tokenized_sequences]

# Pad sequences
padded_sequences = pad_sequence(tokenized_sequences, batch_first=True, padding_value=0)

# Convert labels to tensors
labels = torch.tensor(df['Encoded NAICS Code'].tolist())

# Define the dataset
class NAICSDataset(Dataset):
    def __init__(self, descriptions, labels):
        self.descriptions = descriptions
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.descriptions[idx], self.labels[idx]

# Create the dataset and dataloader
dataset = NAICSDataset(padded_sequences, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model parameters
embedding_dim = 512
hidden_dim = 256
output_dim = len(label_encoder.classes_)  # Number of unique encoded NAICS codes

# Initialize the model, loss function, and optimizer
model = NAICSClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    for descriptions, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(descriptions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for descriptions, labels in dataloader:
        outputs = model(descriptions)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'naics_classifier_model.pth')

# Save the label encoder classes to a CSV file for use during inference
pd.DataFrame({'NAICS Code': label_encoder.classes_}).to_csv('label_encoder_classes.csv', index=False)
