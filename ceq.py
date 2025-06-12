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
from iql import NashQ, DynamicSolver
import time
from scipy.optimize import linprog
from itertools import permutations

class CEQ(NashQ):
    def get_solver(self, env, config):
        self.solver = CESolver(env, config)
        self.solver.set_policy(self.policy, self.target_policy)
        
    def get_runner(self, config):
        self.runner = CEQRunner(self.env, self.policy, self.memory,\
                                    config['eps_start'], config['eps_end'], config['eps_dec'])
        
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
        self.parameter_sharing = config['use-parameter-sharing']
        self.total_gambit_time = 0.0  # 累積總時間的變數
        if not self.dynamic:
            """stationary: solve all state nash and store in a static table"""
            self.strategy = np.zeros((self.env.n_states, self.n_actions**self.n_agents))
            self.static_values = np.zeros((self.env.n_states, self.n_agents))
            for state in range(self.env.n_states):
                
                start_time = time.time()
                
                # 1. solve CE using linear programming
                ce_strategy, ce_payoff = self.solveTSCE(self.env.payoff_matrix[state])
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
            ce_strategy, ce_payoff = self.solveTSCE(dynamic_payoff_matrix[state])
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
        
    def solveTSCE(self, payoff_matrix:np.ndarray):
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
        b_eq = np.array([1], dtype=float)

        # Symmetry constraints: for all permutations within teams
        team_permutations = []

        # 假設 agents 平均分成兩隊（你也可以根據實際隊伍結構調整）
        team1 = list(range(n_agents // 2))
        team2 = list(range(n_agents // 2, n_agents))

        from itertools import permutations

        def add_symmetry_constraints(team, A_eq, b_eq):
            for phi in permutations(team):
                if list(phi) == team:
                    continue  # skip identity
                for idx, joint_action in enumerate(joint_actions):
                    permuted = list(joint_action)
                    for i, pi in zip(team, phi):
                        permuted[i] = joint_action[pi]
                    permuted_idx = joint_actions.index(tuple(permuted))
                    if idx < permuted_idx:  # avoid duplicate
                        constraint = np.zeros(joint_action_count)
                        constraint[idx] = 1
                        constraint[permuted_idx] = -1
                        A_eq = np.vstack([A_eq, constraint])
                        b_eq = np.append(b_eq, 0)

        add_symmetry_constraints(team1, A_eq, b_eq)
        add_symmetry_constraints(team2, A_eq, b_eq)

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
    
if __name__ == '__main__':
    config = {
        'nash-dynamic': False,
        'device': 'cpu',
        'logdir': 'cetest-001'
    }
    def recover_agent_strategies(x: np.ndarray, n_agents: int, n_actions: int) -> np.ndarray:
        """
        從 joint distribution x 還原每個 agent 的 marginal strategy。
        :param x: joint action probability vector, shape = (n_actions ** n_agents,)
        :param n_agents: number of agents
        :param n_actions: number of discrete actions per agent
        :return: ndarray of shape (n_agents, n_actions), 每個 agent 的 marginal 策略
        """
        joint_actions = list(product(range(n_actions), repeat=n_agents))  # 所有 joint action 組合
        agent_strategies = np.zeros((n_agents, n_actions))  # 每位 agent 的 marginal prob 分布
        
        for prob, joint_action in zip(x, joint_actions):
            for i, action in enumerate(joint_action):
                agent_strategies[i, action] += prob  # 累加 marginal 機率
        
        return agent_strategies
    
    for _ in range(1):
        env = TwoTeamSymmetricStochasticEnv(n_states=1, n_agents=4, n_actions=2)
        #env.save(None, config)
        solver = CESolver(env, config)
        print(solver.strategy[0])
        print(recover_agent_strategies(solver.strategy[0], env.n_agents, env.n_actions))
        
#   print(solver.strategy[0])