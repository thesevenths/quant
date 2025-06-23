from pendulum.pendulum_env import PendulumEnv
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

MODEL = A2C(
    policy="MlpPolicy",
    env=make_vec_env(PendulumEnv, n_envs=8),
    n_steps=5,
    gamma=0.9,
    learning_rate=1e-3,
    verbose=0
)

# model = A2C(
#     policy="MlpPolicy",
#     env=make_vec_env(MountainCarEnv, n_envs=8),
#     n_steps=2049,
#     gamma=0.99,
#     learning_rate=1e-4,
#     verbose=0
# )