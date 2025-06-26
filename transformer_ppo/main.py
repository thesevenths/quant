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

def train(load_model=False):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data_path = "btc_daily.csv"
    seq_len = 10
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=1, env_kwargs={'data_path': data_path, 'seq_len': seq_len})

    # Check model existence and load or train
    model_path = 'ppo_trading1.zip'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if load_model and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device)
        logging.info(f"Loaded model from {model_path}")
    else:
        policy_kwargs = {
            'input_dim': 5,
            'd_model': 32,
            'nhead': 4,
            'nlayers': 1,
            'dropout': 0.1,
            'max_len': 5000
        }
        model = PPO(
            policy=TransformerPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            ent_coef=0.01,
            learning_rate=0.001,
            vf_coef=0.75
        )
        model.learn(total_timesteps=600000)
        model.save(model_path)
        logging.info(f"Saved model to {model_path}")
        logging.info("Model structure: %s", model.policy)
        logging.info("Total parameters: %d", sum(p.numel() for p in model.policy.parameters()))

    # Evaluate and collect data for plotting
    log_data = {'steps': [], 'rewards': [], 'prices': [], 'actions': []}
    step = 0
    max_steps = 20000
    obs = env.reset()
    episode_rewards = []
    episode_reward = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    while step < max_steps:
        action, _ = model.predict(obs, deterministic=False)
        action = action[0]  # VecEnv returns array
        action_counts[action] += 1
        obs, reward, done, info = env.step([action])
        episode_reward += reward[0]
        step += 1

        # Log data for plotting (access underlying BTCTradingEnv)
        env_unwrapped = env.envs[0].env  # Unwrap Monitor to get BTCTradingEnv
        log_data['steps'].append(step)
        log_data['rewards'].append(reward[0])
        log_data['prices'].append(env_unwrapped.df['close'].iloc[env_unwrapped.current_step])
        log_data['actions'].append(action)

        if done[0] or info[0].get('TimeLimit.truncated', False):
            episode_rewards.append(episode_reward)
            log_metrics(step, episode_rewards, env_unwrapped.balance)
            logging.info(f"Step {step}, Episode Reward: {episode_reward:.4f}, Balance: {env_unwrapped.balance:.2f}")
            logging.info(f"Action distribution: {action_counts}")
            episode_rewards = []
            episode_reward = 0
            action_counts = {0: 0, 1: 0, 2: 0}
            obs = env.reset()

    # Save log data for plotting
    log_df = pd.DataFrame(log_data)
    log_df.to_csv('training_log_data.csv', index=False)
    logging.info("Saved training log data to training_log_data.csv")

if __name__ == "__main__":
    train(load_model=os.path.exists('ppo_trading1.zip'))