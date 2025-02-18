import json
import numpy as np
import datetime
import os

class Logger():
    def __init__(self):
        self.logEntrys={}
        self.rewards=np.array([])
    
    def add(self, step, reward):   
        self.rewards=np.append(self.rewards,reward)

        if step%100==0:
            mean_reward=np.mean(self.rewards)

            print(f"Step: {step} Reward: {mean_reward}")

            self.rewards=np.array([])

    
    def EpisodeLog(self, totalSteps, EpisodeSteps, reward, episode, loss):
        print(f"Episode: {episode} Steps: {EpisodeSteps} Reward: {reward} Actor Loss: {loss[0]} Critic Loss: {loss[1]}")
        self.logEntrys[episode]={"steps":EpisodeSteps, "reward":reward, "actor_loss":loss[0], "critic_loss":loss[1]}

        if totalSteps%100000==0 and totalSteps>0:
            date=datetime.datetime.now()
            self.save(f"logs/{date.year}-{date.month}-{date.day}", f"{date.hour}-{date.minute}-{date.second}.json")

    def save(self, path, timestamp):
        if not os.path.exists(path):
            os.makedirs(path)

        filepath=f"{path}/{timestamp}"

        with open(filepath, 'w') as f:
            json.dump(self.logEntrys, f)
        self.logEntrys.clear()