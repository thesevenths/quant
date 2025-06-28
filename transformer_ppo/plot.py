import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import os
import sys
# 切换到当前脚本所在目录
os.chdir(sys.path[0])

def plot_btc_rewards(log_file='training_log_data.csv', output_file='btc_reward_plot.png'):
    # Load logged data
    df = pd.read_csv(log_file)
    
    # Calculate cumulative rewards
    df['cumulative_reward'] = df['rewards'].cumsum()
    
    # Normalize price and cumulative reward for comparison
    prices = df['prices'].values
    prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
    cum_rewards = df['cumulative_reward'].values
    cum_rewards = (cum_rewards - cum_rewards.min()) / (cum_rewards.max() - cum_rewards.min() + 1e-8)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['steps'], prices, label='Normalized BTC Price', color='blue')
    plt.plot(df['steps'], cum_rewards, label='Normalized Cumulative Reward', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Normalized Value')
    plt.title('BTC Price vs Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    plot_btc_rewards()