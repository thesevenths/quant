import os
import pandas as pd
import numpy as np
import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import make_env
from model import TransformerPolicy
from utils import log_metrics

import os
import sys
# 切换到当前脚本所在目录
os.chdir(sys.path[0])
print("Current working directory:", os.getcwd())

def calculate_max_drawdown(values):
    max_drawdown = 0.0
    peak = values[0]
    peak_idx = 0
    trough_idx = 0
    for i, value in enumerate(values):
        if value > peak:
            peak = value
            peak_idx = i
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                trough_idx = i
    return max_drawdown, peak_idx, trough_idx


def train(load_model=False):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data_path = "btc_daily.csv"
    seq_len = 10
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1, env_kwargs={'data_path': data_path, 'seq_len': seq_len})

    # Check model existence and load or train
    model_path = 'ppo_trading2.zip'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if load_model and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device)
        logging.info(f"Loaded model from {model_path}")
    else:
        policy_kwargs = {
            'input_dim': 7, # [MODIFIED] 匹配新特征维度（增加了 RSI 和 SMA）
            'd_model': 32,
            'nhead': 4,
            'nlayers': 1,
            'dropout': 0.1,
            'max_len': 5000,
            'optimizer_kwargs': {
                'weight_decay': 1e-4
            }
        }
        model = PPO(
            policy=TransformerPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            ent_coef=0.15,  # [MODIFIED] 增加熵系数以鼓励探索
            learning_rate=0.0003,  # [MODIFIED] 降低学习率以提高稳定性
            vf_coef=2.5,  # [MODIFIED] 增加value loss权重以改善拟合
            n_steps=4096,  # [MODIFIED] 增加每轮步数以获得更稳定梯度
        )
        model.learn(total_timesteps=600000)
        model.save(model_path)
        logging.info(f"Saved model to {model_path}")
        logging.info("Model structure: %s", model.policy)
        logging.info("Total parameters: %d", sum(p.numel() for p in model.policy.parameters()))

   # Evaluate on last 20%, one day at a time
    df = pd.read_csv(data_path)
    total_rows = len(df)
    eval_start_idx = int(total_rows * 0.8)
    eval_df = df.iloc[eval_start_idx:].reset_index(drop=True)
    eval_data_path = "btc_eval_data.csv"
    eval_df.to_csv(eval_data_path, index=False)
    
    log_data = {
        'steps': [], 'rewards': [], 'prices': [], 'actions': [],
        'balance': [], 'holding': [], 'total_asset_value': [], 'datetime': []
    }
    global_step = 0
    episode_rewards = []
    action_counts = {0: 0, 1: 0, 2: 0}
    total_asset_values = []

    try:
        eval_env = make_vec_env(make_env, n_envs=1, env_kwargs={'data_path': 'btc_daily.csv', 'seq_len': seq_len})
        obs = eval_env.reset()

        for day_start in range(seq_len + 1, len(eval_df)):
            # day_df = eval_df.iloc[:day_start].reset_index(drop=True)
            day_df = eval_df.iloc[day_start - seq_len - 1 : day_start].reset_index(drop=True)
            day_data_path = f"btc_eval_day_{day_start}.csv"
            day_df.to_csv(day_data_path, index=False)
            # 根据新数据集跟新obs
            obs = eval_env.envs[0].env.set_observation_window(day_df)

            done = False
            episode_reward = 0
            step = 0
            max_steps = 1

            while step < max_steps and not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action, _ = model.predict(obs_tensor, deterministic=True)  # 返回 (actions, value, log_prob)
                
                # 调试输出
                print(f"action type: {type(action)}, action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")
                
                # 确保 action 是数组
                if not hasattr(action, '__len__') or len(action.shape) == 0:
                    action = action.item()  # 转换为单元素列表
                else:
                    action = action[0] if action.shape[0] == 1 else action  # 取第一个动作
                action_counts[action] += 1
                obs, reward, done, info = eval_env.step([action])
                episode_reward += reward[0]
                step += 1
                global_step += 1

                env_unwrapped = eval_env.envs[0].env
                # 验证 current_step 是否有效;如果 current_step 越界，则重置为 len(env_unwrapped.df) - 1（最后一行的索引）。
                if env_unwrapped.current_step >= len(env_unwrapped.df):
                    print(f"Warning: current_step {env_unwrapped.current_step} out of bounds, resetting to {len(env_unwrapped.df) - 1}")
                    env_unwrapped.current_step = len(env_unwrapped.df) - 1      

                current_price = env_unwrapped.df['close'].iloc[env_unwrapped.current_step]
                holding = info[0].get('holding', 0)  # [MODIFIED] Safely get holding
                total_asset_value = env_unwrapped.balance + holding * current_price
                total_asset_values.append(total_asset_value)

                log_data['steps'].append(global_step)
                log_data['rewards'].append(reward[0])
                log_data['prices'].append(current_price)
                log_data['actions'].append(action)
                log_data['balance'].append(env_unwrapped.balance)
                log_data['holding'].append(holding)
                log_data['total_asset_value'].append(total_asset_value)
                log_data['datetime'].append(env_unwrapped.df['datetime'].iloc[env_unwrapped.current_step])

            episode_rewards.append(episode_reward)
            log_metrics(global_step, episode_rewards, env_unwrapped.balance)
            logging.info(f"Day {day_start}, Step {global_step}, Episode Reward: {episode_reward:.4f}, "
                        f"Balance: {env_unwrapped.balance:.2f}, Holding: {holding:.4f}, "
                        f"Total Asset Value: {total_asset_value:.2f}")
            logging.info(f"Action distribution: {action_counts}")
            action_counts = {0: 0, 1: 0, 2: 0}
            
            if os.path.exists(day_data_path):
                os.remove(day_data_path)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        # Save partial log data
        log_df = pd.DataFrame(log_data)
        log_df.to_csv('eval_log_data.csv', index=False)
        logging.info("Saved partial evaluation log data to eval_log_data.csv")
        raise

    # Calculate max drawdown for total asset value
    if total_asset_values:
        max_drawdown, peak_idx, trough_idx = calculate_max_drawdown(total_asset_values)
        logging.info(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        log_data['max_drawdown'] = [max_drawdown] * len(log_data['steps'])
        log_data['peak_idx'] = [peak_idx] * len(log_data['steps'])
        log_data['trough_idx'] = [trough_idx] * len(log_data['steps'])

    log_df = pd.DataFrame(log_data)
    log_df.to_csv('eval_log_data.csv', index=False)
    logging.info("Saved evaluation log data to eval_log_data.csv")
    return log_df



if __name__ == "__main__":
    train(load_model=os.path.exists('ppo_trading2.zip'))
