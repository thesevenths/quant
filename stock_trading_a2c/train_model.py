import pandas as pd
import sys
sys.path.append('/codes')
from stock_trading_a2c.continues_trading_env import ContinuesTradingEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

data_path = '/stock_trading_a2c/data/510300.SH_20_22.csv'

MODEL = A2C(
    policy="MlpPolicy",
    env=make_vec_env(
                ContinuesTradingEnv,
                n_envs=4,
                env_kwargs={"df": pd.read_csv(data_path)}
            ),
    n_steps=200,
    gamma=0.9,
    learning_rate=1e-4,
    verbose=1
)


if __name__ == '__main__':
    # 训练
    model = MODEL
    model.learn(total_timesteps=10_0000, progress_bar=True)
    model.save("E:/AI_Quant/codes/stock_trading_a2c/save/a2c_model")



