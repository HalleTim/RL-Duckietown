from gym_duckietown.simulator import Simulator
import torch
from env import launchEnv
import config
from logs.logger import Logger
from agent import Agent
import datetime

def trainModel(env, obs, channels, action_dim, max_action, low_action, timestamp):
    
    duckie=Agent(action_dim=action_dim, 
                max_action=max_action,
                low_action=low_action,
                c=channels, 
                buffer_size=config.REPLAY_BUFFER_SIZE, 
                batch_size=config.BATCH_SIZE, 
                lr_actor=config.ACTOR_LR, 
                lr_critic=config.CRITIC_LR, 
                tau=config.TAU, 
                discount=config.DISCOUNT)
    
    done=False
    EpisodeReward=0
    EpisodeSteps=0
    EpisodeNum=0
    logger=Logger(config.EVAL_EPISODE,config.EVAL_STEPS)

    for step in range(int(config.MAX_STEPS)):
        
        if done and step>0:
            done=False
            obs=env.reset()[0]
            loss=duckie.train(EpisodeSteps)
            logger.logEpisode(EpisodeSteps, EpisodeReward, EpisodeNum, loss)
            EpisodeSteps=0
            EpisodeNum+=1
            EpisodeReward=0
            
            

        if (step<config.WARMUP):
            action=env.action_space.sample()
        else:
            action = duckie.select_action(obs)


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

        logger.logSteps(step, reward)


    print("Finished Training")
    print("Saving Model")
     
    duckie.save(f"models/ddpg/{timestamp.year}-{timestamp.month}-{timestamp.day}")
    print("Model Saved")

def evalModel(env, obs, channels, action_dim, max_action, low_action, timestamp):
    duckie=Agent(action_dim=action_dim, 
                max_action=max_action,
                low_action=low_action,
                c=channels,
                trainMode=False)
    duckie.load(f"models/ddpg/{timestamp.year}-{timestamp.month}-{timestamp.day}")
    obs=env.reset()[0]
    done=False
    EvalLogger=Logger(1,1)
    EpisodeNum=0
    EpisodeSteps=0
    EpisodeReward=0 

    for i in range(config.EVAL_STEPS):
        if done:
            obs=env.reset()[0]
            EvalLogger.logEpisode(EpisodeSteps, EpisodeReward, EpisodeNum)
            EpisodeSteps=0
            EpisodeReward=0

        action=duckie.select_action(obs)
        obs, reward, done, truncated, info=env.step(action)
        env.render()

        EvalLogger.logSteps(i, reward)
        EpisodeNum+=1
        EpisodeSteps+=1
        EpisodeReward+=reward



if __name__ == "__main__":
    
    env=launchEnv()
    
    obs = env.reset()[0]
    c=obs.shape[0]

    action_dim = env.action_space.shape[0]
    max_action=env.action_space.high[0]
    low_action=env.action_space.low[0]
    
    timestamp=datetime.datetime.now()
    trainModel(env=env,
               obs=obs,
               channels=c,
               action_dim=action_dim,
               max_action=max_action,
               low_action=low_action,
               timestamp=timestamp)
    
    evalModel(env=env,
               obs=obs,
               channels=c,
               action_dim=action_dim,
               max_action=max_action,
               low_action=low_action,
               timestamp=timestamp)




    

    
    
    

   

        
