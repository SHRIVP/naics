# model.py

import torch
import torch.nn as nn

class NAICSClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(NAICSClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Pass the input through the embedding layer
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embedding_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)

        # Take the output from the last time step (for classification)
        lstm_out = lstm_out[:, -1, :]  # Taking output of the last time step
        # lstm_out shape: (batch_size, hidden_dim)

        # Pass the LSTM output through a fully connected layer
        out = self.fc(lstm_out)
        # out shape: (batch_size, output_dim)

        return self.softmax(out)
