pendulum.model import MODEL


model = MODEL


if __name__ == '__main__':
    # 训练

    model.learn(total_timesteps=100_0000, progress_bar=True)
    model.save("save/a2c_model")




