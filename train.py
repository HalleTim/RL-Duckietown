from gym_duckietown.simulator import Simulator
import gymnasium as gym
from gymnasium import wrappers
import yaml
import gym_duckietown.simulator
from collections import deque
import numpy as np
import torch
import config

def create_env(max_steps):
    env = Simulator(
        seed=123, # random seed
        map_name="loop_empty",
        max_steps= max_steps, # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4, # start close to straight
        full_transparency=True,
        distortion=True,
    )  
    return env

if __name__ == "__main__":
    env=create_env(config.MAX_STEPS)
    env=wrappers.TransformObservation(env, lambda obs: obs.transpose(2,0,1) , env.observation_space)
    state_dim = env.reset()[0].shape
    action_dim = env.action_space.shape[0]
    
    episodes=config.EPISODES
    max_steps=config.MAX_STEPS
