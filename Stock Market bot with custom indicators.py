import gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finta import TA
from gym_anytrading.envs import stocks_env

df = pd.read_csv("D:\Reinforcement Projects\TSLA.csv")
df = df.drop(['Adj Close'], axis=1)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df.sort_values('Date', ascending=True, inplace=True)
df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df.fillna(0, inplace=True)


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features


class MyCustomEnv(stocks_env.StocksEnv):
    _process_data = add_signals


env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(12, 50))

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
#
# env = lambda: env2
# env_vectorized = DummyVecEnv([env])
# model = A2C(policy='MlpPolicy', env=env_vectorized, verbose=1)
# model.learn(total_timesteps=10)
# model.save('Stock_A2C')

env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(80, 580))

obs = env2.reset()
print(obs)
model=A2C.load('Stock_A2C')
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, reward, done, info = env2.step(action)
    print("Amount: {} Action: {} Reward: {}".format(obs, action, reward))
    env2.render()
    if done:
        print("info:- ", info)
        break
