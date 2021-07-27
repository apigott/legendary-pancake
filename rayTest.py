from copy import deepcopy
from numpy import float32
import os
from supersuit import normalize_obs_v0, dtype_v0, color_reduction_v0

import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import PettingZooEnv
from pettingzoo.butterfly import pistonball_v4
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from ray.tune.registry import register_env
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
        self.aspace = self.get_action_space(self.possible_agents)
        self.ospace = self.get_observation_space(self.possible_agents)


    def set_clusters(self):
        clusters = []
        for i in range(self.nclusters):
            clusters += [self.possible_agents[i::self.nclusters]]
        return clusters

    def get_action_space(self, agents):
        aspace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=1*np.ones(1)) for agent in agents}
        return aspace

    def get_observation_space(self, agents):
        ospace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=1*np.ones(1)) for agent in agents}
        return ospace

    def get_spaces(self, agents):
        return self.get_action_space(agents), self.get_observation_space(agents)

class MyEnv(ParallelEnv):
    def __init__(self, grid, tag):
        self.tag = tag
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
        obs = {k: np.array([obs[k]], dtype=np.float32) for k in self.agents}
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
        #print(f"calling step on index {index}")
        # if os.path.isfile(f'temp_{index}.csv'):
        #     df = pd.read_csv(f'temp_{index}.csv')
        # else:
        #     df = self.grid.df
        # if index == 0:
        #     print(df)

        for agent in action_dict.keys():
            self.grid.df.loc[self.grid.df.name==agent, 'observation'] = np.clip(action_dict[agent],-1,1)

        obs = self.state()
        # df.to_csv(f'temp_{index}.csv', index=False)
        #print(self.grid.df)
        return obs, self.get_reward(), self.get_done(), self.get_info()


if __name__ == "__main__":
    """For this script, you need:
    1. Algorithm name and according module, e.g.: "PPo" + agents.ppo as agent
    2. Name of the aec game you want to train on, e.g.: "pistonball".
    3. num_cpus
    4. num_rollouts
    Does require SuperSuit
    """
    alg_name = "SAC"

    nagents = 4
    agents = ['a','b','c','d']
    df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,1) for _ in range(nagents)]})
    grid = MyGrid(df, 1)
    env = MyEnv(grid, "env")
    # Function that outputs the environment you wish to register.
    def env_creator(config):
        nagents = 4
        agents = ['a','b','c','d']
        df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,1) for _ in range(nagents)]})
        grid = MyGrid(df, 1)

        env = MyEnv(grid, "env")
        return env

    num_cpus = 1
    num_rollouts = 2

    # Gets default training configuration and specifies the POMgame to load.
    config = deepcopy(get_trainer_class(alg_name)._default_config)

    # Set environment config. This will be passed to
    # the env_creator function via the register env lambda below.
    config["env_config"] = {}

    # Register env
    register_env("simple_test",
                 lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Configuration for multiagent setup with policy sharing:
    config["multiagent"] = {
        # Setup a single, shared policy for all agents.
        "policies": {k:(None, env.observation_spaces[k], env.action_spaces[k], {}) for k in env.agents},
        # Map all agents to that policy.
        "policy_mapping_fn": lambda agent_id, **kwargs: agent_id,
    }

    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    config["normalize_actions"] = False
    # Fragment length, collected at once from each worker and for each agent!
    config["rollout_fragment_length"] = 30
    # Training batch size -> Fragments are concatenated up to this point.
    config["train_batch_size"] = 200
    # After n steps, force reset simulation
    config["horizon"] = 200
    # Default: False
    config["no_done_at_end"] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.

    # Initialize ray and trainer object
    ray.init(num_cpus=num_cpus + 1)
    trainer = get_trainer_class(alg_name)(env="simple_test", config=config)

    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
    for n in range(5):
        # Train once
        result = trainer.train()
        file_name = trainer.save("tmp/simple_test/ppo")

        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
           ))
    #env = MyEnv()
    ep_reward = 0
    obs = env.reset()
    for _ in range(5):
        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
            action[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
        print(action)
        obs, reward, done, info = env.step(action)
        ep_reward += sum(reward.values())
