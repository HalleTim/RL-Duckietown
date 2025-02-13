import json
import numpy as np
import datetime
import os

class Logger():
    def __init__(self):
        self.logEntrys={}
        self.rewards=np.array([])
        self.distances=np.array([])
        self.losses=np.array([])
        self.wheel_velocities=np.array([])
    
    def add(self, step, reward, dist, wheel_velocities, loss=None):   
        self.rewards=np.append(self.rewards,reward)
        self.distances=np.append(self.distances,dist)
        self.wheel_velocities=np.append(self.wheel_velocities,wheel_velocities)

        if loss is not None:
            self.losses=np.append(self.losses,loss)

        if step%100==0:
            mean_reward=np.mean(self.rewards)
            mean_dist=np.mean(self.distances)
            if np.isnan(self.losses).all():
                mean_loss=None
            else:
                mean_loss=np.mean(self.losses)
            mean_wheel_velocities=np.mean(self.wheel_velocities)

            print(f"Step: {step} Reward: {mean_reward} Dist: {mean_dist} Loss: {mean_dist}")
            self.logEntrys[step]={"reward":mean_dist, "dist":mean_dist, "loss":mean_loss, "wheel_velocities":mean_wheel_velocities}

            self.rewards=np.array([])
            self.distances=np.array([])
            self.losses=np.array([])
            self.wheel_velocities=np.array([])

        if step%10000==0:
            date=datetime.datetime.now()
            self.save(f"logs/{date.year}-{date.month}-{date.day}", f"{date.hour}-{date.minute}-{date.second}.json")
    
    def save(self, path, timestamp):
        if not os.path.exists(path):
            os.makedirs(path)

        filepath=f"{path}/{timestamp}"

        with open(filepath, 'w') as f:
            json.dump(self.logEntrys, f)
        self.logEntrys.clear()