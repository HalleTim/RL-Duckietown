import gymnasium as gym
import numpy as np
from env import launchEnv
import config

from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from TD3_net import customTD3


env = launchEnv()

# The noise objects for TD3
n_actions = env.action_space.shape[-1]

#######DDPG#######
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2* np.ones(n_actions))

#######TD3#######
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2* np.ones(n_actions))

model = TD3("CnnPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=2,
            learning_starts=100,
            buffer_size=config.REPLAY_BUFFER_SIZE, 
            batch_size=100,
            learning_rate=1e-4,
            tensorboard_log="../runs/")

"""model = DDPG("CnnPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=2,
            learning_starts=100,
            buffer_size=int(config.REPLAY_BUFFER_SIZE), 
            batch_size=100,
            learning_rate=1e-4,
            tensorboard_log="../runs/")"""

model.learn(total_timesteps=int(config.MAX_STEPS), log_interval=10, progress_bar=True)
model.save("../runs/DDPG_stable/ddpg_duckietown")

vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("../runs/DDPG_stable/ddpg_duckietown")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")


