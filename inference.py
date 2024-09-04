# inference.py

import torch
import tiktoken
from model import NAICSClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence

# Load the tokenizer (assuming 'cl100k_base' was used)
tokenizer = tiktoken.get_encoding('cl100k_base')

# Load the trained model
vocab_size = tokenizer.n_vocab
embedding_dim = 512  # This should match what was used during training
hidden_dim = 256
output_dim = 19  # Number of unique NAICS codes (this should match your training output_dim)

model = NAICSClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('naics_classifier_model.pth'))
model.eval()

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = pd.read_csv('label_encoder_classes.csv')['NAICS Code'].values

def predict_naics(description, model, tokenizer, label_encoder):
    # Tokenize the input description
    tokenized_description = tokenizer.encode(description)
    tokenized_description = torch.tensor(tokenized_description) # Add batch dimension
    
    # Pad the sequence (if necessary, based on how you trained your model)
    padded_description = pad_sequence([tokenized_description], batch_first=True, padding_value=0)
    print(padded_description.size())
    
    # Make the prediction
    with torch.no_grad():
        output = model(padded_description)
        predicted_index = torch.argmax(output, dim=1).item()
    
    # Convert the predicted index back to the NAICS code
    predicted_naics = label_encoder.inverse_transform([predicted_index])
    
    return predicted_naics[0]

if __name__ == "__main__":
    # Example business description
    new_description = "I perform construction for residential buildings"

    # Predict the NAICS code
    predicted_naics = predict_naics(new_description, model, tokenizer, label_encoder)
    print(f"Predicted NAICS Code: {predicted_naics}")
