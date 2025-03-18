import json
import numpy as np
import datetime
import os

class Logger():
    def __init__(self, stepFrequenzy):
        self.episodeLog={}
        self.stepLong={}
        self.stepShort=np.array([])
        self.startTime=datetime.datetime.now()

        self.stepFrequenzy=stepFrequenzy

    def logSteps(self, step, reward, action=None, info=None):   

        self.stepShort=np.append(self.stepShort,reward)
        print(f"Step: {step} Reward: {reward}")

        if step%self.stepFrequenzy==0:
            mean_reward=np.mean(self.stepShort)
            self.stepShort=np.delete(self.stepShort,0)
            timeSinceStart=datetime.datetime.now()-self.startTime

            if info is not None and 'lane_position' in info['Simulator']:
                self.stepLong[step]={"time": timeSinceStart.seconds, "reward": mean_reward, "lane_pos":info['Simulator']["lane_position"], "wheel_vel":info['Simulator']["wheel_velocities"], "action":action}
            else:
                self.stepLong[step]={"time": timeSinceStart.seconds, "reward": mean_reward}

        if step%100000==0 and step>0:
            self.save(f"logs/eval/{self.startTime.year}-{self.startTime.month}-{self.startTime.day}/trainStepLog", f"{step}.json", self.stepLong)
            self.stepLong.clear()

    
    def logEpisode(self, EpisodeSteps, reward, episode, loss=None):
        timeSinceStart=datetime.datetime.now()-self.startTime
        
        if loss is None:
            print(f"Episode: {episode} Steps: {EpisodeSteps} Reward: {reward}")
            self.episodeLog[episode]={"time": timeSinceStart.seconds, "steps":EpisodeSteps, "reward":reward}
        else:
            print(f"Episode: {episode} Steps: {EpisodeSteps} Reward: {reward} Actor Loss: {loss[0]} Critic Loss: {loss[1]}")


    def save(self, path, timestamp, data):
        if not os.path.exists(path):
            os.makedirs(path)

        filepath=f"{path}/{timestamp}"

        with open(filepath, 'w') as f:
            json.dump(data, f)
        
