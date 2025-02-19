import math
import numpy as np
import torch
from utilis import ReplayBuffer, Normalization, RewardScaling
from envs import *
from modules import MLPAgent, ActorCritic
import random
import pygambit as gbt
from copy import deepcopy
from collections import deque

class BaseRunner():
    def __init__(self, env:NormalFormGame, policy, replay_buffer:ReplayBuffer, algo='IQL'):
        assert env.n_agents == len(policy)
        self.env = env
        if type(self.env.n_actions) == int:
            self.n_actions = self.env.n_actions
        else:
            self.n_actions = self.env.n_actions[0]
        self.policy = policy
        self.n_agents = len(policy)
        self.replay_buffer = replay_buffer
        self.device = policy[0].device  #取得第一個代理策略的運算設備，假設所有代理的策略都在相同的設備上運行
        self.algo = algo

    def store(self, observation, action, reward, done, observation_):
        self.replay_buffer.store(observation, action, reward, done, observation_)

    def run(self):
        raise NotImplementedError

class EGreedyRunner(BaseRunner):
    def __init__(self, env, policy, replay_buffer, eps_start, eps_end, eps_dec, algo='IQL'):
        super().__init__(env, policy, replay_buffer, algo)
        self.epsilon = self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec

    def update_epsilon(self, steps):
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * steps / self.eps_dec)
        
    def run(self, step):
        obs, infos = self.env.reset()
        truncation = termination = False
        total_reward = np.zeros(self.n_agents)
        while (not truncation) and (not termination):
            actions = []
            for i in range(self.env.n_agents):
                if random.random() < self.epsilon:        
                    action = random.sample(range(self.n_actions), 1)[0]
                else:
                    feature = torch.as_tensor(obs[i], dtype=torch.float, device=self.device)
                    action_values = self.policy[i](feature)
                    action = torch.argmax(action_values)
                    action = action.item()
                actions.append(action)

            obs_, reward, termination, truncation, infos = self.env.step(actions)
            total_reward += reward
            self.store(obs, actions, reward, termination, obs_)

            obs = obs_

            step += 1
            self.update_epsilon(step)
            
        return total_reward, step

class NashQRunner(EGreedyRunner):
    def __init__(self, env, policy, replay_buffer, eps_start, eps_end, eps_dec):
        super().__init__(env, policy, replay_buffer, eps_start, eps_end, eps_dec)

    def run(self, step, slover):
        obs, infos = self.env.reset()
        truncation = termination = False
        total_reward = np.zeros(self.n_agents)
        while (not truncation) and (not termination):
            actions = []
            for i in range(self.env.n_agents):
                if random.random() < self.epsilon:        
                    action = random.sample(range(self.n_actions), 1)[0]
                else:
                    "use slover to get best response"
                    strategy = slover(obs[i])
                    strategy = strategy[i]
                    action = np.random.choice([i for i in range(self.n_actions)], size=1, p=strategy)[0]
                actions.append(action)

            obs_, reward, termination, truncation, infos = self.env.step(actions)
            total_reward += reward
            self.store(obs, actions, reward, termination, obs_)

            obs = obs_

            step += 1
            self.update_epsilon(step)
            
        return total_reward, step