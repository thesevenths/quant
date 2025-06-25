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
print(f'device:{device}')

def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences, _ in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions)

# 梯度-based 方法
def analyze_factor_importance_gradient(model, test_loader, feature_names):
    model.eval()
    gradients = []
    for sequences, _ in test_loader:
        sequences = sequences.to(device)
        sequences.requires_grad_()
        outputs = model(sequences)
        grad = torch.autograd.grad(outputs.sum(), sequences)[0]
        gradients.append(grad.cpu().numpy())
    gradients = np.concatenate(gradients)
    feature_importance = np.mean(np.abs(gradients), axis=(0, 1))
    importance = dict(zip(feature_names, feature_importance))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFactor Importance (Gradient-based):")
    for factor, score in sorted_importance:
        print(f"{factor}: {score:.4f}")

# 置换重要性 (Permutation Importance)
def analyze_factor_importance_permutation(model, test_loader, feature_names, criterion):
    model.eval()
    baseline_loss = 0
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        baseline_loss += loss.item()
    baseline_loss /= len(test_loader)

    importance = {}
    for i, feature in enumerate(feature_names):
        permuted_loss = 0
        for sequences, targets in test_loader:
            sequences = sequences.clone()
            sequences[:, :, i] = sequences[:, :, i][torch.randperm(sequences.size(0))]
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            permuted_loss += loss.item()
        permuted_loss /= len(test_loader)
        importance[feature] = permuted_loss - baseline_loss

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFactor Importance (Permutation Importance):")
    for factor, score in sorted_importance:
        print(f"{factor}: {score:.4f}")

# 特征消融 (Feature Ablation)
def analyze_factor_importance_ablation(model, test_loader, feature_names, criterion):
    model.eval()
    baseline_loss = 0
    for sequences, targets in test_loader:
        sequences, targets = sequences.to(device), targets.to(device)
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        baseline_loss += loss.item()
    baseline_loss /= len(test_loader)

    importance = {}
    for i, feature in enumerate(feature_names):
        ablated_loss = 0
        for sequences, targets in test_loader:
            sequences = sequences.clone()
            sequences[:, :, i] = 0  # 将特征设置为 0
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            ablated_loss += loss.item()
        ablated_loss /= len(test_loader)
        importance[feature] = ablated_loss - baseline_loss

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFactor Importance (Feature Ablation):")
    for factor, score in sorted_importance:
        print(f"{factor}: {score:.4f}")

if __name__ == "__main__":
    # Hyperparameters
    seq_length = 30
    batch_size = 32
    num_epochs = 50
    model_dim = 64
    num_heads = 8
    num_layers = 2
    learning_rate = 0.001

    # Load data
    df = dataset.load_data('bitcoin_price.csv')
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    feature_names = ['open', 'high', 'low', 'close', 'volume']
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
        print("No model found...")

    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    sequences, targets = dataset.create_sequences(data_scaled, seq_length)
    split_idx = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, test_targets = targets[:split_idx], targets[split_idx:]

    test_dataset = dataset.TimeSeriesDataset(test_sequences, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Predict on test set
    predictions = predict(model, test_loader)

    # Inverse transform predictions and targets
    dummy_test = np.zeros((len(test_targets), data.shape[1]))
    dummy_test[:, 3] = test_targets
    actual_close_test = scaler.inverse_transform(dummy_test)[:, 3]
    dummy_test[:, 3] = predictions.squeeze()
    predicted_close = scaler.inverse_transform(dummy_test)[:, 3]

    # Inverse transform train targets
    dummy_train = np.zeros((len(train_targets), data.shape[1]))
    dummy_train[:, 3] = train_targets
    actual_close_train = scaler.inverse_transform(dummy_train)[:, 3]

    # Plot actual vs predicted close prices
    plt.figure(figsize=(12, 6))
    # Plot train data
    plt.plot(df['datetime'].iloc[:split_idx], actual_close_train, label='Train Actual Close', color='green')
    # Plot test data
    plt.plot(df['datetime'].iloc[split_idx + seq_length:], actual_close_test, label='Test Actual Close', color='blue')
    plt.plot(df['datetime'].iloc[split_idx + seq_length:], predicted_close, label='Predicted Close', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.title('Actual vs Predicted BTC Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Analyze factor importance using all three methods
    criterion = nn.MSELoss()
    print("\n=== Analyzing Factor Importance ===")
    analyze_factor_importance_gradient(model, test_loader, feature_names)
    analyze_factor_importance_permutation(model, test_loader, feature_names, criterion)
    analyze_factor_importance_ablation(model, test_loader, feature_names, criterion)