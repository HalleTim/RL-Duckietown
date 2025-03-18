import torch
from env import launchEnv
import config
from logs.logger import Logger
from td3_agent import Agent_TD3 as td3
from ddpg_agent import Agent as ddpg
import datetime
import numpy as np
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.noise import NormalActionNoise

def trainModel(env, obs, timestamp):
    
    done=False
    EpisodeReward=0
    EpisodeSteps=0
    EpisodeNum=0
    logger=Logger(config.EVAL_STEPS)

    #ou = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.3)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2* np.ones(n_actions))


    for step in range(int(config.MAX_STEPS)):
        
        if done and step>0:
            done=False
            
            
            if EpisodeNum%config.EVAL_EPISODE==0:
                evalModel(step)

            print(f"Total Steps: {step} Episode: {EpisodeNum} Reward: {EpisodeReward}")

            obs=env.reset()[0]
            EpisodeSteps=0
            EpisodeReward=0
            EpisodeNum+=1

        if (step<config.WARMUP):
            action=env.action_space.sample()
        else:
            action = duckie.select_action(obs)


            if config.EXPL_NOISE != 0:
                #noise = ou.sample((1,2)).cpu().detach().numpy().flatten()
                noise = action_noise()
                action = (action + noise)
                action[0]=np.clip(action[0], -1, 1)
                action[1]=np.clip(action[1], -1, 1)
                #action = np.clip(action, low_action, max_action)
                #noise=np.random.normal(0, config.EXPL_NOISE, size=env.action_space.shape[0])
                


        new_obs, reward, done, info = env.step(action)
        EpisodeSteps+=1
        EpisodeReward+=reward

        if EpisodeSteps>=config.MAX_ENV_STEPS:
            done=True
        
        duckie.storeStep(obs, new_obs, action, reward, done)
        obs=new_obs
        loss=duckie.train(step)

        if len(loss) ==2:
            writer.add_scalar("train/actor_loss", loss[0], step)
            writer.add_scalar("train/critic_loss", loss[1], step)
        else:
            writer.add_scalar("train/critic_loss", loss[0], step)


    print("Finished Training")
    print("Saving Model")
     
    duckie.save(f"../runs/TD3_custom")
    print("Model Saved")

def evalModel(step):

    obs=env.reset()[0]
    done=False
    

    EvalEpisodeSteps=0
    EpisodeReward=0 
    EvalMeanReward=0
    EvalMeanLength=0


    print(f"Evaluating Model for {config.EVAL_LENGTH} episodes")
    for episode in range(1,config.EVAL_LENGTH+1):
        while not done and EvalEpisodeSteps<1500:
            action=duckie.select_action(obs)
            action=np.clip(action, -1, max_action)
            obs, reward, done, EvalInfo=env.step(action)
            env.render()

           
            EvalEpisodeSteps+=1
            EpisodeReward+=reward  
        
        
        obs=env.reset()[0]
        EvalMeanReward+=EpisodeReward
        EvalMeanLength+=EvalEpisodeSteps

        EvalEpisodeSteps=0
        EpisodeReward=0
        done=False  

    writer.add_scalar("rollout/ep_rew_mean", EvalMeanReward/config.EVAL_LENGTH, step)
    writer.add_scalar("rollout/ep_len_mean", EvalMeanLength/config.EVAL_LENGTH, step)

    print(f"**********evaluation complete**********")
    print(f"Mean Reward: {EvalMeanReward/config.EVAL_LENGTH}")
    print(f"time passed: {datetime.datetime.now()-timestamp}")
    print("**************************************")


if __name__ == "__main__":
    
    env=launchEnv()
    
    obs = env.reset()[0]
    c=obs.shape[0]

    action_dim = env.action_space.shape[0]
    max_action=env.action_space.high[0]
    low_action=env.action_space.low[0]
    
    
    timestamp=datetime.datetime.now()
    writer=SummaryWriter(f"../runs/TD3_custom/")

    duckie=td3(action_dim=action_dim, 
            max_action=max_action,
            low_action=low_action,
            c=c, 
            buffer_size=config.REPLAY_BUFFER_SIZE, 
            batch_size=config.BATCH_SIZE, 
            lr_actor=config.ACTOR_LR, 
            lr_critic=config.CRITIC_LR, 
            tau=config.TAU, 
            discount=config.DISCOUNT)
    
    trainModel(env=env,
               obs=obs,
               timestamp=timestamp)



    

    
    
    

   

        
