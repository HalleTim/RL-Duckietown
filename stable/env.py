from os import sys, path
ACTIVE_DIR = path.dirname(path.abspath(__file__))
sys.path.append(path.dirname(ACTIVE_DIR)) 

from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import NotInLane
from gymnasium import wrappers
import gymnasium as gym
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import matplotlib.pyplot as plt

def launchEnv():
    env = Simulator(
        seed=123, # random seed
        map_name="loop_empty_advanced",
        max_steps= 1500, # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4, # start close to straight
        full_transparency=True,
        distortion=True,
    )
    env=integrateWrappers(env)

    return env

def integrateWrappers(env):
    env=wrappers.ResizeObservation(env,(60,80))
    #env=wrappers.GrayscaleObservation(env, True)
    #env=wrappers.RescaleObservation(env, 0,1)
    #env=wrappers.NormalizeObservation(env)
    #env=wrappers.TransformObservation(env, lambda obs: obs.transpose(2,0,1) , env.observation_space)
    #env=wrappers.TransformReward(env, lambda r: -10 if r==-1000 else r+10 if r>0 else r+4)
    env=wrappers.TransformAction(env,lambda a: np.clip(a, 0,0.6), env.action_space)
    #env=wrappers.TransformAction(env,lambda a: [a[0]*0.8, a[1]*0.8], env.action_space)
    #env=MyReward(env)

    
    env=DtRewardPosAngle(env)
    env=DtRewardVelocity(env)


    return env

class MyReward(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(MyReward, self).__init__(env)
        self.orientation_reward = 0.
    
    def reward(self, reward):
        pos = self.unwrapped.cur_pos
        angle = self.unwrapped.cur_angle

        try:
            lp = self.unwrapped.get_lane_pos2(pos, angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            return -1.
        
        if lp.dist<0.1:
            self.orientation_reward= self.unwrapped.speed
        else:
            self.orientation_reward=-1.
        
        return self.orientation_reward
        
    def step(self, action):
        observation, reward, done, truncated, info =self.env.step(action)
        #observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['orientation'] = self.orientation_reward
        return observation, self.reward(reward), done, truncated, info

    def reset(self, **kwargs):
        self.orientation_reward = 0.
        return self.env.reset(**kwargs)
    



class DtRewardPosAngle(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardPosAngle, self).__init__(env)
            # gym_duckietown.simulator.Simulator

        self.max_lp_dist = 0.05
        self.max_dev_from_target_angle_deg_narrow = 10
        self.max_dev_from_target_angle_deg_wide = 50
        self.target_angle_deg_at_edge = 45
        self.scale = 1./2.
        self.orientation_reward = 0.

    def reward(self, reward):
        pos = self.unwrapped.cur_pos
        angle = self.unwrapped.cur_angle
        try:
            lp = self.unwrapped.get_lane_pos2(pos, angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}". format(lp.dist, lp.dot_dir, lp.angle_deg))
        except NotInLane:
            return -10.

        # print("Dist: {:3.2f} | Angle_deg: {:3.2f}".format(normed_lp_dist, normed_lp_angle))
        angle_narrow_reward, angle_wide_reward = self.calculate_pos_angle_reward(lp.dist, lp.angle_deg)
        #logger.debug("Angle Narrow: {:4.3f} | Angle Wide: {:4.3f} ".format(angle_narrow_reward, angle_wide_reward))
        self.orientation_reward = self.scale * (angle_narrow_reward + angle_wide_reward)

        early_termination_penalty = 0.
        # If the robot leaves the track or collides with an other object it receives a penalty
        # if reward <= -1000.:  # Gym Duckietown gives -1000 for this
        #     early_termination_penalty = -10.
        return self.orientation_reward + early_termination_penalty

    def step(self, action):
        observation, reward, done, truncated, info =self.env.step(action)
        #observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['orientation'] = self.orientation_reward
        return observation, self.reward(reward), done, truncated, info

    def reset(self, **kwargs):
        self.orientation_reward = 0.
        return self.env.reset(**kwargs)

    @staticmethod
    def leaky_cosine(x):
        slope = 0.05
        if np.abs(x) < np.pi:
            return np.cos(x)
        else:
            return -1. - slope * (np.abs(x)-np.pi)

    @staticmethod
    def gaussian(x, mu=0., sig=1.):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def calculate_pos_angle_reward(self, lp_dist, lp_angle):
        normed_lp_dist = lp_dist / self.max_lp_dist
        target_angle = - np.clip(normed_lp_dist, -1., 1.) * self.target_angle_deg_at_edge
        reward_narrow = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_narrow)
        reward_wide = 0.5 + 0.5 * self.leaky_cosine(
            np.pi * (target_angle - lp_angle) / self.max_dev_from_target_angle_deg_wide)
        return reward_narrow, reward_wide

class DtRewardVelocity(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardVelocity, self).__init__(env)
        self.velocity_reward = 0.

    def reward(self, reward):
        self.velocity_reward = np.max(self.unwrapped.wheelVels) * 0.25
        if np.isnan(self.velocity_reward):
            self.velocity_reward = 0.
            #logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return reward + self.velocity_reward

    def reset(self, **kwargs):
        self.velocity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['velocity'] = self.velocity_reward
        return observation, self.reward(reward), done, truncated, info
