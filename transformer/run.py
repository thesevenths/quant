import torch
import model
import dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences, _ in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions)

def analyze_factor_importance(model, test_loader, feature_names):
    model.eval()
    with torch.no_grad():
        for sequences, _ in test_loader:
            sequences = sequences.to(device)
            _ = model(sequences)  # Forward pass to populate attn_weights
            if model.attn_weights is not None and model.attn_weights.dim() >= 2:
                attn_weights = model.attn_weights.mean(dim=0).cpu().numpy()  # Average over batch
                feature_scores = attn_weights.mean(axis=0)
                importance = dict(zip(feature_names, feature_scores))
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                print("\nFactor Importance (based on attention weights):")
                for factor, score in sorted_importance:
                    print(f"{factor}: {score:.4f}")
            else:
                print("\nWarning: Attention weights not available. Factor importance analysis skipped.")
            break  # Only need one batch for analysis

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
    # Model parameters
    input_dim = data.shape[1]  # 5 features
    output_dim = 1

    # Model save/load logic
    model_path = 'transformer_model.pth'
    model = model.TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path))
        print(model)
    else:
        print("no model found...")


    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # Create sequences
    sequences, targets = dataset.create_sequences(data_scaled, seq_length)
    # Split into train (80%) and test (20%)
    split_idx = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, test_targets = targets[:split_idx], targets[split_idx:]

    test_dataset = dataset.TimeSeriesDataset(test_sequences, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Predict on test set
    predictions = predict(model, test_loader)

    # Inverse transform predictions and targets
    dummy = np.zeros((len(test_targets), data.shape[1]))
    dummy[:, 3] = test_targets  # Close price column
    actual_close = scaler.inverse_transform(dummy)[:, 3]
    dummy[:, 3] = predictions.squeeze()
    predicted_close = scaler.inverse_transform(dummy)[:, 3]

    # Plot actual vs predicted close prices
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'].iloc[split_idx + seq_length:], actual_close, label='Actual Close', color='blue')
    plt.plot(df['datetime'].iloc[split_idx + seq_length:], predicted_close, label='Predicted Close', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.title('Actual vs Predicted BTC Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Analyze factor importance
    analyze_factor_importance(model, test_loader, feature_names)