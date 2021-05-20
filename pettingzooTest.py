from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo import ParallelEnv
import pandas as pd
import random
import string
import numpy as np
import gym

class MyGrid:
    def __init__(self, dataframe, nclusters):
        self.nclusters = nclusters
        self.df = dataframe
        self.possible_agents = list(self.df['name'])
        self.clusters = self.set_clusters()
        self.cluster = self.clusters[0]
        self.ts = 1

    def set_clusters(self):
        clusters = []
        for i in range(2):
            clusters += [self.possible_agents[i::self.nclusters]]
        return clusters

    def get_action_space(self):
        aspace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in self.cluster}
        return aspace

    def get_observation_space(self):
        ospace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in self.cluster}
        return ospace

    def get_spaces(self):
        self.cluster = self.clusters[self.ts % 2]
        self.ts += 1
        return self.get_action_space(), self.get_observation_space()

class MyEnv(ParallelEnv):
    def __init__(self, grid):
        self.grid = grid
        self.agents = list(self.grid.df['name'])
        self.possible_agents = self.agents[:]
        self.clusters = self.set_clusters()

        self.agents = self.grid.cluster
        self.action_spaces, self.observation_spaces = self.grid.get_spaces()

        self.metadata = {'render.modes': [], 'name':"my_env"}
        self.ts = 0

    def set_other_env(self, env):
        self.other_env = env

    def set_clusters(self):
        clusters = []
        for i in range(2):
            clusters += [self.possible_agents[i::2]]
        return clusters

    def get_action_space(self):
        return {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in self.agents}

    def get_observation_space(self):
        return {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in self.agents}

    def reset(self):
        print('calling reset')
        self.ts = 0
        return self.state()

    def state(self):
        obs = self.grid.df.set_index('name').to_dict()['observation']
        obs = {k: np.array([obs[k]]) for k in self.agents}
        return obs

    def get_reward(self):
        rewards = {agent: float(self.grid.df.loc[self.grid.df.name==agent, 'observation']) for agent in self.agents}
        return rewards

    def get_done(self):
        dones = {agent: False for agent in self.agents}
        return dones

    def get_info(self):
        infos = {agent: {} for agent in self.agents}
        return infos

    def step(self, action_dict):
        print(self.agents, action_dict.keys())
        # assign an env name just so I can keep track
#         # select the agents that will be active this round
#         self.agents = self.possible_agents[(self.ts%2)::2]
        for agent in action_dict.keys():
            self.grid.df.loc[self.grid.df.name==agent, 'observation'] = action_dict[agent]
            self.other_env.grid.df.loc[self.other_env.grid.df.name==agent, 'observation'] = action_dict[agent]

        self.ts += 1

        # for debugging purposes
        print("action", action_dict)
        print(self.grid.df)
        obs = self.state()
        key = list(obs.keys())[0]
        # print(obs[key], obs[key].shape)

        return obs, self.get_reward(), self.get_done(), self.get_info()

if __name__=='__main__':
    import multiprocessing
    import sys
    from pettingzoo.test import parallel_api_test

    # multiprocessing.set_start_method("fork")

    nagents = 5
    agents = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) for _ in range(nagents)]
    df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,2) for _ in range(nagents)]})
    grid = MyGrid(df, 1)

    env = MyEnv(grid)
    env2 = MyEnv(grid)
    env.set_other_env(env2)
    env2.set_other_env(env)
    # parallel_api_test(env)

    env = ss.black_death_v1(env)
    env2 = ss.black_death_v1(env2)

    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env2 = ss.pettingzoo_env_to_vec_env_v0(env2)

    env = ss.concat_vec_envs_v0(env, 2, num_cpus=2, base_class='stable_baselines3')
    env2 = ss.concat_vec_envs_v0(env2, 2, num_cpus=2, base_class='stable_baselines3')

    print('hi')
    envs = [env, env2]
    models = []
    models += [PPO(MlpPolicy, env, verbose=2, gamma=0.999, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1, tensorboard_log="./ppo_test/")]
    models += [PPO(MlpPolicy, env2, verbose=2, gamma=0.999, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1, tensorboard_log="./ppo_test/")]
    for _ in range(10): # timesteps
        for model in models:
            model.learn(1, reset_num_timesteps=False)

    print("XXXXXXXXXXXXX")
    obs = env.reset()
    for _ in range(10):
        for m in range(len(models)):
            # obs = env.state()
            action = models[m].predict(obs)[0]
            obs, reward, done, info = envs[m].step(action)
            obs, reward, done, info = envs[(m+1)%2].step(action)

    # # # multiprocessing pool hangs but I'm not sure where to close it
    # # print('learning done')
