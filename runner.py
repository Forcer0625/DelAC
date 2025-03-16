import math
import numpy as np
import torch
from utilis import ReplayBuffer, Normalization, RewardScaling
from envs import *
from multi_env import MultiEnv
from modules import *
import random
import pygambit as gbt
from copy import deepcopy
from collections import deque

class BaseRunner():
    def __init__(self, env:NormalFormGame, policy, replay_buffer:ReplayBuffer, algo='IQL'):
        self.env = env
        if type(self.env.n_actions) == int:
            self.n_actions = self.env.n_actions
        else:
            self.n_actions = self.env.n_actions[0]
        self.policy = policy
        self.n_agents = self.env.n_agents
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
                    if self.n_agents > len(self.policy) or len(self.policy) == 2:
                        action_values = self.policy[0 if i < self.n_agents//2 else 1](feature)
                    else:
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
    
class OnPolicyRunner():
    def __init__(self, env:MultiEnv, config:dict):
        self.env = env
        self.n_agents = config['n_agents']
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.n_env = config['n_env']
        self.batch_size = config['batch_size']
        self.parameter_sharing = config['use-parameter-sharing']
        self.input_dim = 1
        self.action_dim = config['n_actions']

        self.observations, _ = self.env.reset()
        self.observations = np.transpose(self.observations, (1,0,2))     
        self.dones  = np.zeros((self.n_env), dtype=np.bool_)
        self.truncations = np.zeros((self.n_env), dtype=np.bool_)
        self.actions = np.zeros((self.n_agents, self.n_env), dtype=int)

        # Replay Memory (observation, action, value, reward, a_logp, done)
        self.mb_observations  = np.zeros((self.n_agents, self.batch_size, self.n_env, self.input_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.n_agents, self.batch_size, self.n_env), dtype=np.float32)
        self.mb_values  = np.zeros((self.n_agents, self.batch_size, self.n_env), dtype=np.float32)
        self.mb_rewards = np.zeros((self.n_agents, self.batch_size, self.n_env), dtype=np.float32)
        self.mb_dones   = np.zeros((self.batch_size, self.n_env), dtype=np.bool_)
        self.mb_truncations= np.zeros((self.batch_size, self.n_env), dtype=np.bool_)

        #Reward & length recorder
        self.total_rewards = np.zeros((self.n_agents, self.n_env), dtype=np.float32)
        self.total_len = np.zeros((self.n_env), dtype=np.int32)
        self.reward_buf = deque(maxlen=100)
        self.len_buf = deque(maxlen=100)
    
    def compute_gae(self, rewards, values, dones, last_values, last_dones) -> np.ndarray:
        advs         = np.zeros_like(rewards)
        batch_size   = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_nonterminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - dones[t+1]
                next_values = values[t+1]

            delta   = rewards[t] + self.gamma*next_values*next_nonterminal - values[t]
            advs[t] = last_gae_lam = delta + self.gamma*self.lam*next_nonterminal*last_gae_lam

        return advs + values

    def run(self, policy:list[ActorCritic]):
        self.policy = policy
        self.device = policy[0].device
        episodes = 0
        #1. Run n steps
        #-------------------------------------
        for step in range(self.batch_size):
            self.mb_observations[:, step, :, :]  = self.observations.copy()
            for i in range(self.n_agents):
                observations = torch.from_numpy(self.observations[i]).float().to(self.device)
                actions = torch.from_numpy(self.actions[i]).float().to(self.device)
                action, a_logps, values = self.policy[i](observations, actions)
                action = action.cpu().numpy()
                self.mb_values[i, step, :]  = values.cpu().numpy()
                self.mb_actions[i, step, :] = action
                self.actions[i] = action

            self.observations, rewards, self.dones, self.truncations, _ = self.env.step(np.transpose(self.actions))
            self.observations = np.transpose(self.observations, (1,0,2))
            rewards = np.transpose(rewards, (1,0))

            self.mb_rewards[:, step, :] = rewards
            self.mb_truncations[step, :] = self.truncations
            self.mb_dones[step, :] = self.dones
            last_done = np.logical_or(self.dones, self.truncations)
            episodes += int(last_done.sum())

        last_values = np.zeros((self.n_agents, self.n_env), dtype=np.float32)
        mb_returns = np.zeros((self.n_agents, self.batch_size, self.n_env), dtype=np.float32)
        for i in range(self.n_agents):
            last_values[i] = self.policy[i].critic(torch.from_numpy(self.observations[i]).float().to(self.device), \
                                            torch.from_numpy(self.actions[i]).float().to(self.device)).cpu().numpy()
                
            #2. Compute returns
            #-------------------------------------
            mb_returns[i] = self.compute_gae(self.mb_rewards[i], self.mb_values[i], self.mb_dones, last_values[i], last_done)
        self.record()

        return self.mb_observations.reshape(self.n_agents, self.batch_size*self.n_env, self.input_dim), \
                self.mb_actions.reshape(self.n_agents, self.batch_size*self.n_env), \
                self.mb_values.reshape(self.n_agents, -1), \
                mb_returns.reshape(self.n_agents, -1),\
                episodes
    
    def record(self):
        for i in range(self.batch_size):
            for j in range(self.n_env):
                if self.mb_dones[i, j] or self.mb_truncations[i, j]:
                    # take agent 0's reward as showcase -> [0, i, j]
                    self.reward_buf.append(self.total_rewards[:, j] + self.mb_rewards[:, i, j])
                    self.len_buf.append(self.total_len[j] + 1)
                    self.total_rewards[:, j] = 0
                    self.total_len[j] = 0
                else:
                    self.total_rewards[:, j] += self.mb_rewards[:, i, j]
                    self.total_len[j] += 1

    def get_performance(self):
        if len(self.reward_buf) == 0:
            mean_return = 0
            std_return  = 0
        else:
            mean_return = np.mean(self.reward_buf, axis=0)
            std_return  = np.std(self.reward_buf, axis=0)

        if len(self.len_buf) == 0:
            mean_len = 0
        else:
            mean_len = np.mean(self.len_buf)

        return mean_return, std_return, mean_len
    
    def close(self):
        self.env.close()

class CentralisedOnPolicyRunner(OnPolicyRunner):
    def run(self, actors:list[Actor], critic:CentralisedCritic):
        self.device = critic.device
        episodes = 0
        #1. Run n steps
        #-------------------------------------
        for step in range(self.batch_size):
            self.mb_observations[:, step, :, :]  = self.observations.copy()
            for i in range(self.n_agents):
                observations = torch.from_numpy(self.observations[i]).float().to(self.device)
                action, a_logs = actors[i](observations)
                action = action.cpu().numpy()
                self.mb_actions[i, step, :] = action
                self.actions[i] = action

            actions = torch.from_numpy(self.actions).float().to(self.device)
            actions = actions.transpose(1 ,0)
            values = critic(observations, actions) # (n_env, 2)
            values = np.transpose(values.cpu().numpy())
            self.mb_values[:self.n_agents//2, step, :]  = values[0]
            self.mb_values[self.n_agents//2:, step, :]  = values[1]

            self.observations, rewards, self.dones, self.truncations, _ = self.env.step(np.transpose(self.actions))
            self.observations = np.transpose(self.observations, (1,0,2))
            rewards = np.transpose(rewards, (1,0))

            self.mb_rewards[:, step, :] = rewards
            self.mb_truncations[step, :] = self.truncations
            self.mb_dones[step, :] = self.dones
            last_done = np.logical_or(self.dones, self.truncations)
            episodes += int(last_done.sum())

        last_values = np.zeros((self.n_agents, self.n_env), dtype=np.float32)
        mb_returns = np.zeros((self.n_agents, self.batch_size, self.n_env), dtype=np.float32)

        values = critic(torch.from_numpy(self.observations[0]).float().to(self.device),
                        torch.from_numpy(self.actions).float().to(self.device).transpose(1, 0)).cpu().numpy()
        values = np.transpose(values)
        for i in range(self.n_agents):
            last_values[i] = values[0 if i < self.n_agents//2 else 1]
            mb_returns[i] = self.compute_gae(self.mb_rewards[i], self.mb_values[i], self.mb_dones, last_values[i], last_done)

        self.record()

        return self.mb_observations.reshape(self.n_agents, self.batch_size*self.n_env, self.input_dim), \
                self.mb_actions.reshape(self.n_agents, self.batch_size*self.n_env), \
                self.mb_values.reshape(self.n_agents, -1), \
                mb_returns.reshape(self.n_agents, -1),\
                episodes