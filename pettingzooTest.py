# RL imports
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import supersuit as ss
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
import gym

# other useful stuff
import pandas as pd
import random, string
import numpy as np
from copy import deepcopy, copy

# this represents the CityLearn grid
class MyGrid:
    def __init__(self, dataframe, nclusters):
        self.nclusters = nclusters
        self.df = dataframe # analogous to the pandapower network (dataframes)
        self.possible_agents = list(self.df['name'])
        self.clusters = self.set_clusters()
        self.ts = 0

    def set_clusters(self):
        clusters = []
        for i in range(self.nclusters): # let every other agent be on the opposing team
            clusters += [self.possible_agents[i::self.nclusters]]
        return clusters

    def get_action_space(self, agents):
        # return the action space of each agent {agent: gym.space(data)}
        aspace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in agents}
        return aspace

    def get_observation_space(self, agents):
        # return the observation space of each agent {agent: gym.space(data)}
        ospace = {agent: gym.spaces.Box(low=-1*np.ones(1), high=np.ones(1)) for agent in agents}
        return ospace

class MyEnv(ParallelEnv):
    def __init__(self, grid):
        # initialize env with one team of agents
        self.agents = grid.clusters.pop()
        self.possible_agents = self.agents[:]
        # and get the corresponding actionspaces
        self.action_spaces = grid.get_action_space(self.agents)
        self.observation_spaces = grid.get_observation_space(self.agents)

        self.metadata = {'render.modes': [], 'name':"my_env"} # gym stuff, not used
        self.ts = 0 # if we want to track num. steps these agents have taken

    def reset(self):
        # reset the environment to something randomized
        for agent in self.agents:
            self.grid.df.loc[self.grid.df.name==agent, 'observation'] = np.random.uniform(0,1)
        return self.state()

    def state(self):
        # return the observed state {agent: np.array(data)}
        obs = {agent: np.array(float(self.grid.df.loc[self.grid.df.name==agent, 'observation'])) for agent in self.agents}
        return obs

    def get_reward(self):
        # return reward {agent: float(data)}
        # in this case, reward = state = action
        rewards = {agent: float(self.grid.df.loc[self.grid.df.name==agent, 'observation']) for agent in self.agents}
        return rewards

    def get_done(self):
        # environment never ends, return {agent: bool(data)}
        dones = {agent: False for agent in self.agents}
        return dones

    def get_info(self):
        # no additional info to track
        infos = {agent: {} for agent in self.agents}
        return infos

    def step(self, action_dict):
        # input: actions to take = {agent: action}
        # output: dictionaries of format {agent: data} for obs, reward, done, info
        for agent in action_dict.keys():
            self.grid.df.loc[self.grid.df.name==agent, 'observation'] = action_dict[agent]

        print(self.grid.df) # check how the dataframe evolves
        self.grid.ts += 1
        self.ts += 1
        return self.state(), self.get_reward(), self.get_done(), self.get_info()

if __name__=='__main__':
    # create a dumb dataframe so we can see how the agents interact
    agents = ['a','b','c','d'] # 4 agents, 2 teams
    nteams = 2 # number of teams
    df = pd.DataFrame({'name':agents,'observation':[np.random.uniform(0,2) for _ in range(len(agents))]})
    grid = MyGrid(df,nteams)

    # create some parallel environments
    teams = [MyEnv(grid), MyEnv(grid)]
    # cast them to pettingzoo envs (so we can train multiple agents on them)
    teams = [ss.pettingzoo_env_to_vec_env_v0(team) for team in teams]
    # copy environment and stack it so we can use PPO and get [nenvs] sets of data at once
    # each team trains on 2 environments "in parallel"
    nenvs = 2
    teams = [ss.concat_vec_envs_v0(team, nenvs, num_cpus=1, base_class='stable_baselines3') for team in teams]

    # set the grid after initialization, this way the object is shared across env instances
    # this also saves RAM by only copying the grid when we need it
    # (otherwise concat_vec_envs_v0 will copy it nteams*nenvs times)
    grids = [deepcopy(grid) for n in range(nenvs)]
    for team in teams:
        for n in range(nenvs):
            team.venv.vec_envs[n].par_env.grid = grids[n]

    # create some stable baselines models
    # batch_size = nenvs*nsteps and MUST be greater than 1
    models = [PPO(MlpPolicy, team, verbose=2, gamma=0.999, batch_size=2, n_steps=1, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=4, clip_range=0.2, clip_range_vf=1) for team in teams]

    # train the models
    for ts in range(100):
        for model in models: # each timestep alternate through models to take turns
            model.learn(1, reset_num_timesteps=False)

    # reset the models to test them
    obss = [team.reset() for team in teams]
    for ts in range(5): # test on 5 timesteps
        for m in range(len(models)): # again, alternate through models

            # get the current observation from the perspective of the active team
            # this can probably be cleaned up
            foo = []
            for e in range(nenvs):
                bar = list(teams[m].venv.vec_envs[e].par_env.state().values())
                foo += bar # may need additional logic to pad state/obs spaces if they aren't identical

            foo = np.vstack(foo)
            obss[m] = np.vstack(foo)

            action = models[m].predict(obss[m])[0] # send it to the SB model to select an action
            obss[m], reward, done, info = teams[m].step(action) # update environment
