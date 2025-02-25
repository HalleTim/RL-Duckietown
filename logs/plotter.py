import matplotlib.pyplot as plt
import numpy as np
import json
import os

class  Plotter():
    def __init__(self):
        pass
    
    def load(self, path):
        json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
        json_files.sort()
        result={}
        for file in json_files:
            with open(f"{path}/{file}") as f:
                data=json.load(f)
                result={**result, **data}
        return result
    
    def updatePlot(self):
        episodes=self.load("logs/rewards/2025-2-23")
        x=[]
        y=[]
        timePassed=0
        for key in episodes.keys():
            x.append(key)
            y.append(episodes[key]["reward"])
            if episodes[key]["time"]>timePassed:
                timePassed=episodes[key]["time"]
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.figure(figsize=(50,25))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.locator_params(axis='both', nbins=6)
        plt.plot(x,y)

        
        plt.savefig("logs/rewards/2025-2-23/rewards.png")
        
plot=Plotter()

plot.updatePlot()