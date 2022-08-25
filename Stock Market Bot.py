import gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("D:\Reinforcement Projects\TSLA.csv")
df=df.drop(['Adj Close'],axis=1)
df['Date']=pd.to_datetime(df['Date'])
df=df.set_index('Date')
#
# env=gym.make('stocks-v0',df=df,frame_bound=(5,100),window_size=5)
#
# state=env.reset()
# while True:
#     action=env.action_space.sample()
#     n_state,reward,done,info=env.step(action)
#     if done:
#         print("info:- ",info)
#         break
# plt.figure(figsize=(15,4))
# plt.cla()
# env.render_all()
# plt.show()

# env=lambda: gym.make('stocks-v0',df=df,frame_bound=(5,100),window_size=5)
# env_vectorized=DummyVecEnv([env])
# model=A2C(policy='MlpPolicy',env=env_vectorized,verbose=1)
# model.learn(total_timesteps=500000)
# model.save('Stock_A2C')

env=gym.make('stocks-v0',df=df,frame_bound=(90,150),window_size=5)
obs=env.reset()
model=A2C.load('Stock_A2C')
while True:
    obs=obs[np.newaxis,...]
    action,_states=model.predict(obs)
    obs,reward,done,info=env.step(action)
    print("Amount: {} Action: {} Reward: {}".format(obs, action, reward))
    env.render()
    if done:
        print("info:- ",info)
        break
