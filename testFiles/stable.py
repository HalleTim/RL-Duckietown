import gymnasium as gym
import numpy as np
from env import launchEnv
import config

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common import env_checker

env = launchEnv()

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("CnnPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=2,
            learning_starts=config.WARMUP)
model.learn(total_timesteps=10000, log_interval=10)
model.save("td3_pendulum")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = TD3.load("td3_pendulum")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")