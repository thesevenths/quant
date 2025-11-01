# main.py
import logging
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import make_env
from model import TransformerPolicy   # 保持你原来使用的 TransformerPolicy

import os 
import sys # 切换到当前脚本所在目录 
os.chdir(sys.path[0]) 
print("Current working directory:", os.getcwd())

def calculate_max_drawdown(values: list):
    max_dd = 0.0
    peak = values[0]
    peak_idx = 0
    trough_idx = 0
    for i, v in enumerate(values):
        if v > peak:
            peak = v
            peak_idx = i
        else:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
                trough_idx = i
    return max_dd, peak_idx, trough_idx

def train(load_model=False,
          model_path='ppo_trading3.zip',
          data_path='btc_daily.csv',
          seq_len=10,
          initial_balance=10_000_000,
          **env_kwargs):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建向量化环境
    env = make_vec_env(lambda: make_env(data_path=data_path, seq_len=seq_len,
                                        initial_balance=initial_balance, **env_kwargs),
                       n_envs=1)

    if load_model and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device)
        logging.info(f"Loaded model from {model_path}")
    else:
        policy_kwargs = {
            'input_dim': len(env.envs[0].env.use_features),
            'd_model': 32,
            'nhead': 4,
            'nlayers': 2,
            'dropout': 0.1,
            'max_len': 5000,
            'optimizer_kwargs': {'weight_decay': 1e-3}
        }
        model = PPO(policy=TransformerPolicy,
                    env=env,
                    verbose=1,
                    device=device,
                    ent_coef=0.05, # entropy coefficient:encourages exploration by penalizing certainty
                    learning_rate=0.0005,
                    vf_coef=2,
                    n_steps=2048,
                    policy_kwargs=policy_kwargs)
        
        model.learn(total_timesteps=600_000)
        model.save(model_path)
        logging.info(f"Saved model to {model_path}")
        logging.info("Model structure: %s", model.policy)
        logging.info("Total parameters: %d", sum(p.numel() for p in model.policy.parameters()))

    # ===== 评估部分 =====
    df = pd.read_csv(data_path)
    total_rows = len(df)
    eval_start_idx = int(total_rows * 0.8)
    eval_df = df.iloc[eval_start_idx:].reset_index(drop=True)

    eval_log = {
        'steps': [], 'rewards': [], 'actions': [], 'balance': [], 'holding': [],
        'total_asset_value': [], 'datetime': []
    }
    env_eval = make_vec_env(lambda: make_env(data_path=data_path, seq_len=seq_len,
                                              initial_balance=initial_balance, **env_kwargs),
                           n_envs=1)
    obs = env_eval.reset()

    total_asset_values = []
    for t in range(len(eval_df) - seq_len):
        action, _ = model.predict(obs, deterministic=True)
        # obs, reward, done, truncated, info = env_eval.step(action)
        obs, rewards, dones, infos = env_eval.step(action)
        env_inst = env_eval.envs[0].env
        dt = env_inst.df['datetime'].iloc[env_inst.current_step]
        total_asset = env_inst.balance + env_inst.holding * env_inst.df['close'].iloc[env_inst.current_step]

        eval_log['steps'].append(t)
        eval_log['rewards'].append(rewards[0])
        eval_log['actions'].append(int(action[0]))
        eval_log['balance'].append(env_inst.balance)
        eval_log['holding'].append(env_inst.holding)
        eval_log['total_asset_value'].append(total_asset)
        eval_log['datetime'].append(dt)
        total_asset_values.append(total_asset)

        if dones:
            break

    dd, peak_idx, trough_idx = calculate_max_drawdown(total_asset_values)
    eval_df_log = pd.DataFrame(eval_log)
    eval_df_log['max_drawdown'] = dd
    eval_df_log['peak_idx'] = peak_idx
    eval_df_log['trough_idx'] = trough_idx

    eval_df_log.to_csv('eval_log_data.csv', index=False)
    logging.info(f"Max Drawdown: {dd * 100:.2f}%")

    # 额外评价指标：胜率、平均持仓天数、交易次数、盈亏比
    # 假定 info 或日志包含持仓天数／交易盈亏等，可在此后续计算

    return eval_df_log

if __name__ == "__main__":
    # 可以通过 env_kwargs 传止损／止盈／滑点／杠杆等参数
    train(load_model=os.path.exists('ppo_trading3.zip'),
          data_path='btc_daily.csv',
          seq_len=10,
          initial_balance=10_000_000,
          trading_fee=0.001,
          slippage_pct=0.0005,
          max_hold_days=10,
          use_leverage=1.0,
          stop_loss_pct=-0.05,
          take_profit_pct=0.10)
