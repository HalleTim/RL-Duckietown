from env import launchEnv
from td3_agent import Agent_TD3 as td3
from ddpg_agent import Agent as ddpg
import datetime
import config



def launchTest():
    obs=env.reset()[0]
    done=False
    total_reward=0

    while True:
        action=duckie.select_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward+=reward

        if done:
            obs=env.reset()[0]
            done=False
            print(total_reward)
            total_reward=0
            print(f"Episode reward: {total_reward}")



env=launchEnv()

obs = env.reset()[0]
c=obs.shape[0]

action_dim = env.action_space.shape[0]
max_action=env.action_space.high[0]
low_action=env.action_space.low[0]

timestamp=datetime.datetime.now()

duckie=ddpg(action_dim=action_dim, 
        max_action=max_action,
        low_action=low_action,
        c=c, 
        buffer_size=config.REPLAY_BUFFER_SIZE, 
        batch_size=config.BATCH_SIZE, 
        lr_actor=config.ACTOR_LR, 
        lr_critic=config.CRITIC_LR, 
        tau=config.TAU, 
        discount=config.DISCOUNT)

duckie.load("../runs/DDPG_custom")
launchTest()