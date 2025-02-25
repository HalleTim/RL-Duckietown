from gym_duckietown.simulator import Simulator
import gymnasium as gym
from gymnasium import wrappers
import torch
import numpy as np

import config
from logs.logger import Logger
from agent import Agent


def create_env(max_steps):
    env = Simulator(
        seed=123, # random seed
        map_name="loop_empty",
        max_steps= 500001, # we don't want the gym to reset itself
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
    env=wrappers.ResizeObservation(env,(120,160))
    env=wrappers.NormalizeObservation(env)
    env=wrappers.TransformObservation(env, lambda obs: obs.transpose(2,0,1) , env.observation_space)
    env=wrappers.TransformReward(env, lambda r: -10 if r==-1000 else r+10 if r>0 else r+4)
    env=wrappers.TransformAction(env,lambda a:[a[0]*0.8,a[1]*0.8], env.action_space)
    
    #env=wrappers.ClipAction(env)

    obs = env.reset()[0]
    c=obs.shape[0]
    action_dim = env.action_space.shape[0]
    max_action=env.action_space.high[0]

    duckie=Agent(action_dim, max_action,c , config.REPLAY_BUFFER_SIZE, config.BATCH_SIZE, config.ACTOR_LR, config.CRITIC_LR, config.GAMMA, config.TAU, config.DISCOUNT)

    done=False
    EpisodeReward=0
    EpisodeSteps=0
    EpisodeNum=0
    logger=Logger()

    for step in range(int(config.MAX_STEPS)):
        
        if done and step>0:
            done=False
            obs=env.reset()[0]
            loss=duckie.train(EpisodeSteps)
   
            logger.EpisodeLog(step, EpisodeSteps, EpisodeReward, EpisodeNum, loss)
            EpisodeSteps=0
            EpisodeNum+=1
            EpisodeReward=0
            
            

        if (step<config.WARMUP):
            action=env.action_space.sample()
        else:
            action = duckie.select_action(obs)
            #action=torch.clamp(action,min=0, max=env.action_space.high)
            """if config.EXPL_NOISE != 0:
                action = (action + np.random.normal(
                    0,
                    config.EXPL_NOISE,
                    size=env.action_space.shape[0])
                          ).clip(env.action_space.low, env.action_space.high)"""


        new_obs, reward, done, truncated, info = env.step(action)
        EpisodeSteps+=1
        EpisodeReward+=reward

        if EpisodeSteps>=config.MAX_ENV_STEPS:
            done=True
        env.render()
        
        duckie.storeStep(obs, new_obs, action, reward, done)
        obs=new_obs

        logger.add(step, reward)
    print("Finished Training")
    print("Saving Model")
    duckie.save("models/ddpg/")
    print("Model Saved")

        
