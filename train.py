from gym_duckietown.simulator import Simulator
import torch
from env import launchEnv
import config
from logs.logger import Logger
from td3_agent import Agent_TD3 as td3
import datetime
import numpy as np
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

def trainModel(env, obs, channels, action_dim, max_action, low_action, timestamp):
    
    done=False
    EpisodeReward=0
    EpisodeSteps=0
    EpisodeNum=0
    logger=Logger(config.EVAL_EPISODE,config.EVAL_STEPS)
    ou = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.3, base_scale=0.25)
    epsilon=1
    epsilon_decay=1e-6
    for step in range(int(config.MAX_STEPS)):
        
        if done and step>0:
            done=False
            loss=duckie.train(EpisodeSteps)
            epsilon-=epsilon_decay*EpisodeSteps
            
            if EpisodeNum%config.EVAL_EPISODE==0 and EpisodeNum>0:
                evalModel(step)
                
            print(f"Episode: {EpisodeNum} Steps: {EpisodeSteps} Reward: {EpisodeReward} Actor Loss: {loss[0]} Critic Loss: {loss[1]}")

            obs=env.reset()[0]
            EpisodeSteps=0
            EpisodeReward=0
            EpisodeNum+=1

        if (step<config.WARMUP):
            action=env.action_space.sample()
        else:
            action = duckie.select_action(obs)


            if config.EXPL_NOISE != 0:
                noise = ou.sample((1,2)).cpu().detach().numpy().flatten()*epsilon
                action = (action + noise)
                action[0]=np.clip(action[0], 0, 1)
                action[1]=np.clip(action[1], -1, 1)
                #action = np.clip(action, low_action, max_action)
                #noise=np.random.normal(0, config.EXPL_NOISE, size=env.action_space.shape[0])
                


        new_obs, reward, done, info = env.step(action)
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
     
    duckie.save(f"logs/eval/{timestamp.year}-{timestamp.month}-{timestamp.day}/model")
    print("Model Saved")

def evalModel(step):

    obs=env.reset()[0]
    done=False
    EvalLogger=Logger(1,1)

    EvalEpisodeSteps=0
    EpisodeReward=0 
    EvalMeanReward=0
    duckie.evalMode()

    print(f"Evaluating Model for {config.EVAL_LENGTH} episodes")
    for episode in range(1,config.EVAL_LENGTH+1):
        while not done and EvalEpisodeSteps<500:
            action=duckie.select_action(obs)
            action=np.clip(action, -1, max_action)
            obs, reward, done, EvalInfo=env.step(action)
            env.render()

            EvalLogger.logSteps(EvalEpisodeSteps, reward, action.tolist(), EvalInfo)
           
            EvalEpisodeSteps+=1
            EpisodeReward+=reward
        evalTimestamp=datetime.datetime.now()     
        EvalLogger.save(f"logs/eval/{timestamp.year}-{timestamp.month}-{timestamp.day}/{step}/steps/Episode-{episode}", f"{evalTimestamp.hour}-{evalTimestamp.minute}-{evalTimestamp.second}.json", EvalLogger.stepLong) 
        EvalLogger.stepLong.clear()
        
        if done or EvalEpisodeSteps>=499:
            obs=env.reset()[0]
            EvalLogger.logEpisode(EvalEpisodeSteps, float(EpisodeReward), episode)
            EvalMeanReward+=EpisodeReward
            EvalEpisodeSteps=0
            EpisodeReward=0
            done=False
            
    evalTimestamp=datetime.datetime.now()     
    
    
    EvalLogger.save(f"logs/eval/{timestamp.year}-{timestamp.month}-{timestamp.day}/{step}/episodes", f"{evalTimestamp.hour}-{evalTimestamp.minute}-{evalTimestamp.second}.json", EvalLogger.episodeLog)
    EvalLogger.episodeLog.clear()
    duckie.trainM()
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
               channels=c,
               action_dim=action_dim,
               max_action=max_action,
               low_action=low_action,
               timestamp=timestamp)



    

    
    
    

   

        
