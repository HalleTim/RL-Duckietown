import json
import numpy as np
import datetime
import os
import time

class Logger():
    def __init__(self, episodeFrequency, stepFrequenzy):
        self.episodeLog={}
        self.stepLong={}
        self.stepShort=np.array([])
        self.startTime=time.time()

        self.episodeFrequency=episodeFrequency
        self.stepFrequenzy=stepFrequenzy

    def logSteps(self, step, reward):   

        self.stepShort=np.append(self.stepShort,reward)
        print(f"Step: {step} Reward: {reward}")

        if step%self.stepFrequenzy==0:
            mean_reward=np.mean(self.stepShort)
            self.stepShort=np.delete(self.stepShort,0)
            timeSinceStart=time.time()-self.startTime

            self.stepLong[step]={"time": timeSinceStart, "reward": mean_reward}

        if step%100000==0 and step>0:
            date=datetime.datetime.now()
            self.save(f"logs/steps/{date.year}-{date.month}-{date.day}", f"{date.hour}-{date.minute}-{date.second}steps.json", self.rewardsLong)
            self.stepLong.clear()

    
    def logEpisode(self, EpisodeSteps, reward, episode, loss=None):
        if loss is None:
            print(f"Episode: {episode} Steps: {EpisodeSteps} Reward: {reward}")
            self.episodeLog[episode]={"time": timeSinceStart, "steps":EpisodeSteps, "reward":reward}
            
        timeSinceStart=time.time()-self.startTime
        print(f"Episode: {episode} Steps: {EpisodeSteps} Reward: {reward} Actor Loss: {loss[0]} Critic Loss: {loss[1]}")
        

        if episode>0 and episode%self.episodeFrequency==0:
            date=datetime.datetime.now()
            self.episodeLog[episode]={"time": timeSinceStart, "steps":EpisodeSteps, "reward":reward, "actor_loss":loss[0].item(), "critic_loss":loss[1].item()}
            self.save(f"logs/rewards/{date.year}-{date.month}-{date.day}", f"{date.hour}-{date.minute}-{date.second}.json", self.episodeLog)
            self.episodeLog.clear()


    def save(self, path, timestamp, data):
        if not os.path.exists(path):
            os.makedirs(path)

        filepath=f"{path}/{timestamp}"

        with open(filepath, 'w') as f:
            json.dump(data, f)
        