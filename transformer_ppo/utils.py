import numpy as np

def normalize(array):
    return (array - np.mean(array)) / (np.std(array) + 1e-8)

def log_metrics(step, rewards, balance, filename="training_log.txt"):
    with open(filename, 'a') as f:
        f.write(f"Step {step}: Mean Reward = {np.mean(rewards):.4f}, Balance = {balance:.2f}\n")