import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import dataset
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

if __name__ == "__main__":
    # Hyperparameters
    seq_length = 30  # Use past 30 days
    batch_size = 32
    num_epochs = 50
    model_dim = 64
    num_heads = 8
    num_layers = 2
    learning_rate = 0.001

    # Load data
    df = dataset.load_data('btc_daily.csv')
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    feature_names = ['open', 'high', 'low', 'close', 'volume']

    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    sequences, targets = dataset.create_sequences(data_scaled, seq_length)

    # Split into train (80%) and test (20%)
    split_idx = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, test_targets = targets[:split_idx], targets[split_idx:]

    # Create datasets and loaders
    train_dataset = dataset.TimeSeriesDataset(train_sequences, train_targets)
    # test_dataset = dataset.TimeSeriesDataset(test_sequences, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_dim = data.shape[1]  # 5 features
    output_dim = 1

    # Model save/load logic
    model_path = 'transformer_model.pth'
    model = model.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

    if os.path.exists(model_path):
        print("model already existed...")
        # model.load_state_dict(torch.load(model_path))
    else:
        print("Training new model...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, criterion, optimizer, num_epochs)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")