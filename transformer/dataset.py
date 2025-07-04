from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx],
                                                                                    dtype=torch.float32)

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df = df.sort_values('datetime')
    return df

# Create sequences for time series
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length, 3]  # Close price is the 4th column (index 3)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)