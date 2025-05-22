import torch
import torch.nn as nn
import numpy as np
from itertools import product
from modules import MLPAgent
from utilis import ReplayBuffer
from runner import CEQRunner
from envs import *
from copy import deepcopy
import pygambit as gbt
from torch.utils.tensorboard import SummaryWriter
from iql import NashQ, DynamicSolver, NashQwLinearRegression
import time
from scipy.optimize import linprog

class CEQ(NashQ):
    def get_solver(self, env, config):
        self.solver = CESolver(env, config)
        self.solver.set_policy(self.policy, self.target_policy)
        
    def get_runner(self, config):
        self.runner = CEQRunner(self.env, self.policy, self.memory,\
                                    config['eps_start'], config['eps_end'], config['eps_dec'])
        
class CEQwLinearRegression(NashQwLinearRegression):
    def get_sub_policy(self):
        self.subgame_algo = []
        for l in range(self.n_subgames):
            subgame_config = deepcopy(self.config)
            subgame_config['logdir'] = subgame_config['logdir'] + '-subgame-'+str(l+1)
            self.subgame_algo.append(CEQ(self.env.subgames[l], subgame_config))
            
    def get_answer(self):
        if self.env.n_states == 1:
            config = deepcopy(self.config)
            config['nash-dynamic'] = False
            self.answer = CESolver(self.env, config)

class CESolver(DynamicSolver):
    def __init__(self, env:StochasticGame, config):
        self.env = env
        self.n_agents = self.env.n_agents
        if type(self.env.n_actions) == int:
            self.n_actions = self.env.n_actions
        else:
            self.n_actions = self.env.n_actions[0]
        self.dynamic = config['nash-dynamic']
        self.device = config['device']
        self.total_gambit_time = 0.0  # 累積總時間的變數
        if not self.dynamic:
            """stationary: solve all state nash and store in a static table"""
            self.strategy = np.zeros((self.env.n_states, self.n_actions**self.n_agents))
            self.static_values = np.zeros((self.env.n_states, self.n_agents))
            for state in range(self.env.n_states):
                
                start_time = time.time()
                
                # 1. solve CE using linear programming
                ce_strategy, ce_payoff = self.solveCE(self.env.payoff_matrix[state])
                # 2. get probability and store
                self.strategy[state] = ce_strategy
                # 3. calculate expected payoff for every state given nash
                self.static_values[state] = ce_payoff
            
                end_time = time.time()
                self.total_gambit_time += (end_time - start_time)
        else:
            """dynamic: slove nash based on state and Q-network"""
            self.saving = False
    
    def __call__(self, state):
        """strategy: (n_actions**n_agents)"""
        if (not self.dynamic) or self.saving:
            return np.squeeze(self.strategy[state])

        self.strategy = np.zeros((self.env.n_states, self.n_actions**self.n_agents))
        self.static_values = np.zeros((self.env.n_states, self.n_agents))
        dynamic_payoff_matrix = self.from_qvalues()
        for state in range(self.env.n_states):
            
            start_time = time.time()
            
            # 1. solve CE using linear programming
            ce_strategy, ce_payoff = self.solveCE(dynamic_payoff_matrix[state])
            # 2. get probability and store
            self.strategy[state] = ce_strategy
            # 3. calculate expected payoff for every state given nash
            self.static_values[state] = ce_payoff
        
            end_time = time.time()
            self.total_gambit_time += (end_time - start_time)
            
        self.saving = True
        return self(state)
    
    def to_joint_action(self, index):
        """Convert flat index to joint action [a0, a1, ..., an], where a0 is agent 0's action"""
        joint_action = []
        for _ in range(self.n_agents):
            joint_action.append(index % self.n_actions)
            index //= self.n_actions
        return joint_action  # agent 0 is the least significant

    def from_joint_action(self, joint_action):
        """Convert joint action [a0, a1, ..., an] to flat index"""
        index = 0
        for i in reversed(range(self.n_agents)):
            index = index * self.n_actions + joint_action[i]
        return index
    
    def solveCE(self, payoff_matrix:np.ndarray):
        n_agents = self.n_agents
        n_actions = self.n_actions
        joint_action_count = n_actions ** n_agents

        # Flatten payoff_matrix to (n_joint_actions, n_agents)
        joint_actions = list(product(range(n_actions), repeat=n_agents))
        flat_payoffs = np.zeros((joint_action_count, n_agents))
        for idx, joint_action in enumerate(joint_actions):
            flat_payoffs[idx] = payoff_matrix[joint_action]

        # Objective: maximize sum of expected payoffs => -c for linprog (minimization)
        c = -np.sum(flat_payoffs, axis=1)

        # Equality constraint: sum of probabilities == 1
        A_eq = np.ones((1, joint_action_count))
        b_eq = [1]

        # Inequality constraints for CE condition
        A_ub = []
        b_ub = []

        for i in range(n_agents):
            for ai in range(n_actions):
                for a_hat_i in range(n_actions):
                    if ai == a_hat_i:
                        continue
                    coeffs = np.zeros(joint_action_count)
                    for idx, a in enumerate(joint_actions):
                        if a[i] == ai:
                            a_hat = list(a)
                            a_hat[i] = a_hat_i
                            a_hat_idx = joint_actions.index(tuple(a_hat))
                            coeffs[idx] += flat_payoffs[idx][i]
                            coeffs[idx] -= flat_payoffs[a_hat_idx][i]
                    A_ub.append(coeffs)
                    b_ub.append(0)

        bounds = [(0, 1) for _ in range(joint_action_count)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            ce_strategy = result.x
            ce_payoff = ce_strategy @ flat_payoffs  # shape: (n_agents,)
            return ce_strategy, ce_payoff
        else:
            raise ValueError("Linear programming failed to find a solution.")
    
    def get_static_expected_payoff(payoff_matrix:np.ndarray, ce_strategy:np.ndarray) -> np.ndarray:
        n_agents = payoff_matrix.shape[-1]
        n_actions = payoff_matrix.shape[0]
        expected_payoffs = np.zeros(n_agents)

        # 生成所有可能的行為組合，這裡使用 (0, 0, ..., 0) 到 (1, 1, ..., 1)
        all_actions = list(product([i for i in range(n_actions)], repeat=n_agents))

        # 遍歷每個行為組合
        for actions in all_actions:
            # 計算該行為組合的機率
            actions_ = list(actions)
            actions_.reverse()
            index = 0
            for i in range(n_agents):
                index += (actions_[i]*pow(n_actions, n_agents-i-1))
            prob = ce_strategy[index]

            # 從收益矩陣中獲取該行為組合的收益
            payoffs = payoff_matrix[actions]
            
            # 根據該行為組合的機率加權收益
            expected_payoffs += prob * payoffs

        return expected_payoffs