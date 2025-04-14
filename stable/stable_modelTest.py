import gymnasium as gym
import numpy as np
from env import launchEnv
import config
import time

from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.spaces import Box


env = launchEnv()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2* np.ones(n_actions))

#policy netwoks
model = TD3("CnnPolicy", 
            env, 
            action_noise=action_noise, 
            verbose=2,
            learning_starts=100,
            buffer_size=int(config.REPLAY_BUFFER_SIZE), 
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

vec_env = model.get_env()

#load model
model = TD3.load("runs/simple/TD3_stable/td3_duckietown")
#model = DDPG.load("runs/simple/DDPG_stable/ddpg_duckietown")

obs = vec_env.reset()
dones=False

total_reward=0
episode_reward=0
episode_mean_speed=0
episode_mean_orientation=0
episode_num=0

mean_speed=0
mean_orientation=0
survival_time=0
steps=0
time_stamp=time.time()

#model evaluation
while episode_num < config.EVAL_EPISODE:
    if dones == True:      
        total_reward += episode_reward
        episode_mean_speed += mean_speed/steps
        episode_mean_orientation += mean_orientation/steps
        survival_time += time.time() - time_stamp
        
        steps = 0
        mean_speed = 0
        mean_orientation = 0
        episode_reward = 0
        time_stamp = time.time()
        episode_num += 1


    action, _states = model.predict(obs)
    #action = np.clip(action, 0, 0.5)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

    mean_orientation+=info[0]['custom_rewards']['orientation']
    mean_speed+=info[0]['custom_rewards']['speed']
    episode_reward+=rewards
    steps+=1

print(f"Total reward: {total_reward/config.EVAL_EPISODE}")
print(f"Mean speed: {episode_mean_speed/config.EVAL_EPISODE}")
print(f"Mean orientation: {episode_mean_orientation/config.EVAL_EPISODE}")
print(f"Survival time: {survival_time/config.EVAL_EPISODE}")

 

