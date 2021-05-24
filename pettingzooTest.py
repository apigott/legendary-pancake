from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo import ParallelEnv
import pandas as pd
import random
import string
import numpy as np
import gym
import os

import multiprocessing
import sys
from pettingzoo.test import parallel_api_test

from copy import deepcopy, copy

class MyGrid:
    def __init__(self, dataframe, nclusters):
        self.nclusters = nclusters
        self.df = dataframe
        self.possible_agents = list(self.df['name'])
        self.clusters = self.set_clusters()
        self.cluster = self.clusters[0]
        self.file = "temp.csv"
        self.df.to_csv(self.file, index=False)
        self.ts = 0

    def set_clusters(self):
        clusters = []
        for i in range(2):
            clusters += [self.possible_agents[i::self.nclusters]]
        return clusters

    def get_action_space(self, agents):
        aspace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in agents}
        return aspace

    def get_observation_space(self, agents):
        ospace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in agents}
        return ospace

    def get_spaces(self, agents):
        return self.get_action_space(agents), self.get_observation_space(agents)

class MyEnv(ParallelEnv):
    def __init__(self, grid):
        self.grid = grid
        self.agents = list(self.grid.df['name'])

        self.agents = self.grid.clusters.pop()
        self.possible_agents = self.agents[:]
        self.action_spaces, self.observation_spaces = self.grid.get_spaces(self.agents)

        self.metadata = {'render.modes': [], 'name':"my_env"}
        self.file='temp.csv'
        self.ts = 0

    def reset(self):
        print('calling reset')
        return self.state()

    def state(self):
        # if os.path.isfile(f'temp_{index}.csv'):
        #     df = pd.read_csv(f'temp_{index}.csv')
        # else:
        #     df = self.grid.df
        # df = pd.read_csv(f'temp_{index}.csv')
        obs = self.grid.df.set_index('name').to_dict()['observation']
        obs = {k: np.array([obs[k]]) for k in self.agents}
        return obs

    def get_reward(self):
        # df = pd.read_csv(f'temp_{index}.csv')
        rewards = {agent: float(self.grid.df.loc[self.grid.df.name==agent, 'observation']) for agent in self.agents}
        return rewards

    def get_done(self):
        dones = {agent: False for agent in self.agents}
        return dones

    def get_info(self):
        infos = {agent: {} for agent in self.agents}
        return infos

    def step(self, action_dict):
        for agent in action_dict.keys():
            self.grid.df.loc[self.grid.df.name==agent, 'observation'] = action_dict[agent]

        obs = self.state()
        # df.to_csv(f'temp_{index}.csv', index=False)
        # print(self.grid.ts, hex(id(self.grid.ts)))

        print(self.grid.df)
        self.grid.ts += 1
        return obs, self.get_reward(), self.get_done(), self.get_info()

nagents = 4
agents = ['a','b','c','d']
df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,2) for _ in range(nagents)]})
grid = MyGrid(df,2)

# print("grid ", hex(id(grid)))
# x = grid
# print("assignment ", hex(id(x)))
# print("copy of ", hex(id(copy(grid))))
# print("copy of ", hex(id(deepcopy(grid))))
envs = [MyEnv(grid), MyEnv(grid)]
envs = [ss.pettingzoo_env_to_vec_env_v0(env) for env in envs]

nenvs = 2
envs = [ss.concat_vec_envs_v0(env, nenvs, num_cpus=1, base_class='stable_baselines3') for env in envs]

for env in envs[1:]:
    for n in range(nenvs):
        env.venv.vec_envs[n].par_env.grid = envs[0].venv.vec_envs[n].par_env.grid

models = [PPO(MlpPolicy, env, verbose=2, gamma=0.999, batch_size=2, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1) for env in envs]

for _ in range(100):
    for model in models:
        model.learn(1, reset_num_timesteps=False)

obss = [env.reset() for env in envs]
for _ in range(5):
    for m in range(len(models)):
        obss[m] = [np.concatenate(list(envs[m].venv.vec_envs[i].par_env.state().values())) for i in range(len(envs[m].venv.vec_envs))]
        obss[m] = np.concatenate(obss[m])
        obss[m] = obss[m].reshape(-1,1)

        action = models[m].predict(obss[m])[0]
        obss[m], reward, done, info = envs[m].step(action)
