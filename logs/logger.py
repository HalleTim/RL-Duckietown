import json
import numpy as np
import datetime

class Logger():
    def __inin__(self):
        self.logEntrys={}
        self.rewards=np.array([])
        self.distances=np.array([])
        self.losses=np.array([])
    
    def add(self, step, reward, dist, loss=None):
        self.rewards.append(reward)
        self.distances.append(dist)
        self.losses.append(loss)

        if step%100==0:
            mean_reward=np.mean(self.rewards)
            mean_dist=np.mean(self.distances)
            mean_loss=np.mean(self.losses)

            print(f"Step: {step} Reward: {mean_reward} Dist: {mean_dist} Loss: {mean_dist}")
            self.logEntrys[step]={"reward":mean_dist, "dist":mean_dist, "loss":mean_loss}
        elif step%1000==0:
            date=datetime.datetime.now()
            self.save(f"/logging/logs/{date}log.json" %date)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.logEntrys, f)
        self.logEntrys.clear()