from env import launchEnv
from td3_agent import Agent_TD3 as td3
from ddpg_agent import Agent as ddpg
import time
import config
import numpy as np



def launchTest():
    obs=env.reset()[0]
    
    done=False

    total_reward=0
    episode_reward=0
    episode_mean_speed=0
    episode_mean_orientation=0

    mean_speed=0
    mean_orientation=0
    survival_time=0

    EpisodeNum=0
    steps=0
    


    time_stamp=time.time()

    while EpisodeNum<config.EVAL_EPISODE:
        action=duckie.select_action(obs)
        #action=np.clip(action, 0, 0.5)
        obs, reward, done, info = env.step(action)

        mean_orientation+=info['custom_rewards']['orientation'] 
        mean_speed+=env.unwrapped.speed
        episode_reward+=reward
        
        steps+=1
        env.render()

        if done:
            obs=env.reset()[0]
            done=False
            episode_mean_speed+=mean_speed/steps
            episode_mean_orientation+=mean_orientation/steps
            survival_time+=time.time()-time_stamp


            EpisodeNum+=1
            total_reward+=episode_reward
            
            steps=0
            mean_speed=0
            episode_reward=0
            mean_orientation=0
            time_stamp=time.time()
            
            
    print(f"Mean total reward: {total_reward/config.EVAL_EPISODE},  mean speed: {episode_mean_speed/config.EVAL_EPISODE}, mean orientation: {episode_mean_orientation/config.EVAL_EPISODE}, survival time: {survival_time/config.EVAL_EPISODE}")

env=launchEnv()

obs = env.reset()[0]
c=obs.shape[0]

action_dim = env.action_space.shape[0]
max_action=env.action_space.high[0]
low_action=env.action_space.low[0]



duckie=ddpg(action_dim=action_dim, 
        max_action=max_action,
        low_action=low_action,
        c=c, 
        buffer_size=config.REPLAY_BUFFER_SIZE, 
        batch_size=config.BATCH_SIZE, 
        lr_actor=config.ACTOR_LR, 
        lr_critic=config.CRITIC_LR, 
        tau=config.TAU, 
        discount=config.DISCOUNT)

duckie.load("runs/simple/DDPG_custom")
launchTest()
