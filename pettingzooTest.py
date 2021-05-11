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

class MyEnv(ParallelEnv):
    def __init__(self, grid):
        self.grid = grid
        self.agents = list(self.grid['name'])
        self.possible_agents = self.agents[:]

        self.observation_spaces = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in self.agents}
        self.action_spaces = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in self.agents}
        print(self.observation_spaces, self.action_spaces)

        self.metadata = {'render.modes': []}
        self.metadata['name'] = "my_env"
        self.ts = 0

    def shape_obs(self, obs_dict):
        for k, v in obs_dict.items():
            # if v is a dataframe entry then it's a float, cast to np.array()
            try:
                v.shape
            except:
                obs_dict[k] = np.array([v])
        return obs_dict

    def reset(self):
        obs = self.grid.set_index('name').to_dict()['observation']
        obs = self.shape_obs(obs) # dict is key: <type float> change to key: <type ndarray>
        return obs

    def get_reward(self):
        rewards = {}
        for agent in self.agents:
            rewards[agent] = float(self.grid.loc[self.grid.name==agent, 'observation'])
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
            self.grid.loc[self.grid.name==agent, 'observation'] = action_dict[agent]
        obs = self.grid.set_index('name').to_dict()['observation']
        print(self.ts)
        self.ts += 1
        obs = self.shape_obs(obs)
        return obs, self.get_reward(), self.get_done(), self.get_info()

    def state(self):
        obs = self.grid.set_index('name').to_dict()['observation']
        obs = self.shape_obs(obs)
        return obs

if __name__=="__main__":
    from pettingzoo.test import api_test, parallel_api_test
    from pettingzoo.mpe import simple_push_v2
    import random
    import string
    from stable_baselines3.ppo import MlpPolicy
    from stable_baselines3 import PPO
    import pandas as pd
    import multiprocessing
    print('done with imports')
    multiprocessing.set_start_method("fork")

    agents = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5)) for _ in range(2)]
    df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,2) for _ in range(2)]})
    env = MyEnv(df)

    # parallel_api_test(env)

    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 4, num_cpus=4, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=2, gamma=0.999, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1, tensorboard_log="./ppo_test/")
    model.learn(10)
    print('done learning')
