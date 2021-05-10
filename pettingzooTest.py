from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
# from petting_bubble_env_continuous import PettingBubblesEnvironment
from pettingzoo import ParallelEnv
import pandas as pd
import random
import string
import numpy as np
import gym

# class BigEnv(AECEnv, agents):
    # def __init__(self):
    #     self.nclusters = 2
    #     self.clusters = [agents[i::2] for i in range(self.nclusters)]
    #
    #     self.envs = []
    #     for cluster in self.cluster:
    #         self.envs += [SmallEnv(cluster)]
    #
    # def reset(self):
    #     return obs
    #
    # def get_reward(self):
    #     return reward
    #
    # def get_done(self):
    #     return done
    #
    # def get_info(self):
    #     return info
    #
    # def step(self, action):
    #     return obs, self.get_reward(), self.get_done(), self.get_info()
    #
    # def state(self):
    #     return obs

class MyEnv(ParallelEnv):
    def __init__(self, agents, grid):
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.grid = grid
        self.observation_spaces = dict(zip(self.agents, [gym.spaces.Box(low=0, high=2, shape=(1,))]))
        self.action_spaces = dict(zip(self.agents, [gym.spaces.Box(low=-1, high=1, shape=(1,))]))

        self.metadata = {'render.modes': []}
        self.metadata['name'] = "my_env"

    def shape_obs(self, obs_dict):
        for k, v in obs_dict.items():
            try:
                v.shape
            except:
                obs_dict[k] = np.array([v])
        return obs_dict

    def reset(self):
        obs = self.grid.set_index('name').to_dict()['observation']
        obs = self.shape_obs(obs)
        return obs

    def get_reward(self):
        rewards = {}
        for agent in self.agents:
            rewards[agent] = self.grid.loc[self.grid.name==agent, 'observation']
        return rewards

    def get_done(self):
        dones = {}
        for agent in self.agents:
            dones[agent] = False
        return dones

    def get_info(self):
        infos = {}
        for agent in self.agents:
            infos[agent] = []
        return infos

    def step(self, action_dict):
        for agent in self.agents:
            self.grid.loc[self.grid.name==agent, 'observation'] = np.random.randint(3)
        obs = self.grid.set_index('name').to_dict()['observation']
        obs = self.shape_obs(obs)
        return obs, self.get_reward(), self.get_done(), self.get_info()

    def state(self):
        obs = self.grid.set_index('name').to_dict()['observation']
        obs = self.shape_obs(obs)
        return obs

if __name__=="__main__":
    from pettingzoo.test import api_test, parallel_api_test
    import random
    import string
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3 import PPO
    from stable_baselines3.sac.policies import MlpPolicy
    from stable_baselines3 import SAC
    import pandas as pd
    import multiprocessing
    print('done with imports')
    multiprocessing.set_start_method("fork")

    agents = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) for _ in range(4)]
    df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,2) for _ in range(4)]})
    env = MyEnv(agents, df)

    print('modifying env')
    env = ss.pad_observations_v0(env)
    env = ss.black_death_v1(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 4, num_cpus=4, base_class='stable_baselines3')

    print('making model')
    model = PPO(MlpPolicy, env)
    # for agent in agents()s
    print('done with this')
