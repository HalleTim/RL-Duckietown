import json
import numpy as np
import datetime
import os
import time

class Logger():
    def __init__(self):
        self.logEntrys={}
        self.rewardsLong={}
        self.rewardsShort=np.array([])
        self.StartTime=time.time()

    def add(self, step, reward):   

        self.rewardsShort=np.append(self.rewardsShort,reward)
        print(f"Step: {step} Reward: {reward}")
        if step%1000==0:
            mean_reward=np.mean(self.rewardsShort)
            self.rewardsShort=np.array([])
            timeSinceStart=time.time()-self.StartTime

            self.rewardsLong[step]={"time": timeSinceStart, "reward": mean_reward}

        if step%100000==0 and step>0:
            date=datetime.datetime.now()
            self.save(f"logs/steps/{date.year}-{date.month}-{date.day}", f"{date.hour}-{date.minute}-{date.second}steps.json", self.rewardsLong)
            self.rewardsLong.clear()

    
    def EpisodeLog(self, totalSteps, EpisodeSteps, reward, episode, loss):
        
        timeSinceStart=time.time()-self.StartTime
        print(f"Episode: {episode} Steps: {EpisodeSteps} Reward: {reward} Actor Loss: {loss[0]} Critic Loss: {loss[1]}")
        

        if episode>0 and episode%1000==0:
            date=datetime.datetime.now()
            self.logEntrys[episode]={"time": timeSinceStart, "steps":EpisodeSteps, "reward":reward, "actor_loss":loss[0], "critic_loss":loss[1]}
            self.save(f"logs/rewards/{date.year}-{date.month}-{date.day}", f"{date.hour}-{date.minute}-{date.second}.json", self.logEntrys)
            self.logEntrys.clear()

    def save(self, path, timestamp, data):
        if not os.path.exists(path):
            os.makedirs(path)

        filepath=f"{path}/{timestamp}"

        with open(filepath, 'w') as f:
            json.dump(data, f)
        