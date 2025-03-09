from rltools import TD3
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

seed = 0xf00d
def env_factory():
    env = gym.make("Pendulum-v1")
    env = RescaleAction(env, -1, 1)
    env.reset(seed=seed)
    return env

sac = TD3(env_factory)
state = sac.State(seed)

finished = False
while not finished:
    finished = state.step()