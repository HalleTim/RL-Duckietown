from gym_duckietown.simulator import Simulator
import gymnasium as gym
from gymnasium import wrappers
import torch
import config
from logs.logger import Logger
from agent import Agent

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
    if torch.cuda.is_available():
        cuda=True
    else:
        cuda=False
    print(f"Is unsing cuda: %s" %cuda)
    
    env=create_env(config.MAX_STEPS)
    env=wrappers.TransformObservation(env, lambda obs: obs.transpose(2,0,1) , env.observation_space)

    obs = env.reset()[0]
    c=obs.shape[0]
    action_dim = env.action_space.shape[0]

    duckie=Agent(action_dim, c , 4, config.REPLAY_BUFFER_SIZE, config.BATCH_SIZE, config.ACTOR_LR, config.CRITIC_LR, config.GAMMA, config.TAU)

    done=False
    RewardEpisode=0
    logger=Logger()
    for step in range(config.MAX_STEPS):
        
        if (step<config.RANDOM_STEPS):
            action=env.action_space.sample()
        else:
            action=duckie.select_action(obs)

        new_obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        duckie.storeStep(obs, new_obs, action, reward, done)

        dist_center=info["Simulator"]["lane_position"]["dist"]
        wheel_velocities=info["Simulator"]["wheel_velocities"]
        
        if done and step>0:
            loss=duckie.train()
            done=False
            obs=env.reset()[0]
        else:
            logger.add(step, reward, dist_center, wheel_velocities)
            obs=new_obs

        
