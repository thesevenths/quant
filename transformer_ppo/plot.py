import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def plot_btc_rewards(data_path='btc_daily.csv', eval_log_file='eval_log_data.csv', output_file='btc_eval_plot1.png'):
    # Load full dataset and evaluation log
    full_df = pd.read_csv(data_path)
    eval_df = pd.read_csv(eval_log_file)
    
    # Ensure datetime is in correct format
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])
    eval_df['datetime'] = pd.to_datetime(eval_df['datetime'])
    
    # Calculate cumulative rewards
    eval_df['cumulative_reward'] = eval_df['rewards'].cumsum()
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot BTC price (left y-axis)
    ax1.plot(full_df['datetime'], full_df['close'], label='BTC Price', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('BTC Price (USD)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    # Plot total asset value (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(eval_df['datetime'], eval_df['total_asset_value'], label='Total Asset Value', color='green', linestyle='-', marker='o', markersize=4)
    ax2.set_ylabel('Total Asset Value (USD)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Plot cumulative reward (scaled to total asset value)
    reward_scaled = eval_df['cumulative_reward'] * (eval_df['total_asset_value'].max() - eval_df['total_asset_value'].min()) / (eval_df['cumulative_reward'].max() - eval_df['cumulative_reward'].min() + 1e-8) + eval_df['total_asset_value'].min()
    ax2.plot(eval_df['datetime'], reward_scaled, label='Scaled Cumulative Reward', color='orange', linestyle=':')
    
    # Add max drawdown shading
    max_drawdown = eval_df['max_drawdown'].iloc[0]
    peak_idx = int(eval_df['peak_idx'].iloc[0])
    trough_idx = int(eval_df['trough_idx'].iloc[0])
    if peak_idx < trough_idx and peak_idx < len(eval_df['datetime']) and trough_idx < len(eval_df['datetime']):
        ax2.axvspan(eval_df['datetime'].iloc[peak_idx], eval_df['datetime'].iloc[trough_idx], color='red', alpha=0.3, label=f'Max Drawdown {max_drawdown * 100:.2f}%')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('BTC Price, Total Asset Value, and Scaled Cumulative Reward over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    plot_btc_rewards()