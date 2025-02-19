import torch
import torch.nn as nn
import numpy as np
from itertools import product
from modules import MLPAgent
from utilis import ReplayBuffer
from runner import *
from envs import *
from copy import deepcopy
import pygambit as gbt
from torch.utils.tensorboard import SummaryWriter
from qmix import QMIX
import time

class IQL(QMIX):
    def __init__(self, env:NormalFormGame, config):
        self.env = env
        self.n_agents = self.env.n_agents
        if type(self.env.n_actions) == int:
            self.n_actions = self.env.n_actions
        else:
            self.n_actions = self.env.n_actions[0]
        observation, _ = self.env.reset()

        self.batch_size = config['batch_size']
        self.memory_size = config['memory_size']
        self.memory = ReplayBuffer(self.memory_size, observation[0].shape, self.n_agents)

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.device = config['device']
        self.loss = nn.MSELoss()

        self.get_policy(observation)

        self.get_runner(config)

        self.logger = SummaryWriter('./runs/'+config['logdir'])
        self.infos = []
        self.config = config

    def get_policy(self, observation):
        self.policy = []
        self.target_policy = []
        self.optimizer = []
        for i in range(self.n_agents):
            self.policy.append(MLPAgent(observation[0].reshape(-1).shape[0], self.n_actions, device=self.device))
            self.target_policy.append(deepcopy(self.policy[i]))
            self.optimizer.append(torch.optim.Adam(self.policy[i].parameters(), lr=self.lr)) 

    def get_runner(self, config):
        self.runner = EGreedyRunner(self.env, self.policy, self.memory,\
                                    config['eps_start'], config['eps_end'], config['eps_dec'])

    def update(self, ):
        observations, actions, rewards,\
            dones, observations_ = self.memory.sample(self.batch_size)
        
        
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)#.view(self.n_agents, *observations[0][0].shape)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device).view(-1)
        
        observations_= torch.as_tensor(observations_, dtype=torch.float32, device=self.device)#.view(-1, *observations_[0][0].shape)

        total_loss = 0.0
        for i in range(self.n_agents):
            action_values = self.policy[i](observations[:,i]).reshape(-1, self.n_actions)
            action_values = action_values.gather(1, actions[:,i].unsqueeze(1))
            action_values = action_values.reshape(-1)

            # double-q
            with torch.no_grad():
                estimate_action_values = self.policy[i](observations_[:,i]).reshape(-1, self.n_actions)
                next_action = torch.max(estimate_action_values, dim=1).indices
                next_action_values = self.target_policy[i](observations_[:,i]).reshape(-1, self.n_actions)
                next_action_values = next_action_values.gather(1, next_action.unsqueeze(1))
                next_action_values = next_action_values.reshape(-1)

            # calculate loss
            target = rewards[:,i] + self.gamma * (1 - dones) * next_action_values
            loss = self.loss(action_values, target.detach())

            # optimize
            self.optimizer[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy[i].parameters(), 10)
            self.optimizer[i].step()
            total_loss+=loss.item()

        return total_loss/self.n_agents

    def sync(self, ):
        for i in range(self.n_agents):
            target_net_weights = self.target_policy[i].state_dict()
            q_net_weights = self.policy[i].state_dict()
            for key in q_net_weights:
                target_net_weights[key] = q_net_weights[key]*self.tau + target_net_weights[key]*(1-self.tau)
            self.target_policy[i].load_state_dict(target_net_weights)

    def hard_sync(self):
        for i in range(self.n_agents):
            self.target_policy[i].load_state_dict(self.policy[i].state_dict())

    def save_model(self, path=None):
        if path is None:
            path = './model/'+self.config['logdir']
        models = {}
        for i in range(self.n_agents):
            models['agent'+str(i)] = self.policy[i].state_dict()
        torch.save(models, path)
        
    def load_model(self, path=None):
        if path is None:
            path = './model/'+self.config['logdir']
        models = torch.load(path)
        for i in range(self.n_agents):
            self.policy[i].load_state_dict(models['agent'+str(i)])  
            self.target_policy[i].load_state_dict(models['agent'+str(i)])

class NashQ(IQL):
    def __init__(self, env, config):
        self.dynamic = config['nash-dynamic']
        super().__init__(env, config)
        self.get_solver(env, config)

    def get_policy(self, observation):
        self.policy = []
        self.target_policy = []
        self.optimizer = []
        for i in range(self.n_agents):
            self.policy.append(MLPAgent(observation[0].reshape(-1).shape[0], self.n_actions, device=self.device))
            if self.dynamic:
                self.target_policy.append(deepcopy(self.policy[i]))
            self.optimizer.append(torch.optim.Adam(self.policy[i].parameters(), lr=self.lr))

    def get_runner(self, config):
        self.runner = NashQRunner(self.env, self.policy, self.memory,\
                                    config['eps_start'], config['eps_end'], config['eps_dec'])

    def get_solver(self, env, config):
        self.solver = DynamicSolver(env, config)
        if self.dynamic:
            self.solver.set_policy(self.policy, self.target_policy)

    def update(self):
        observations, actions, rewards,\
            dones, observations_ = self.memory.sample(self.batch_size)
        
        values = self.solver.values(np.array(observations[:,0], dtype=np.int64))#np.squeeze(self.static_value(observations[:,0])) # (n_batches, n_agents)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device).view(-1)
        
        observations_= torch.as_tensor(observations_, dtype=torch.float32, device=self.device)#.view(-1, *observations_[0][0].shape)

        total_loss = 0.0
        if self.dynamic:
            self.solver.saving = False
        for i in range(self.n_agents):
            action_values = self.policy[i](observations[:,i]).reshape(-1, self.n_actions)
            action_values = action_values.gather(1, actions[:,i].unsqueeze(1))
            action_values = action_values.reshape(-1)

            # calculate loss
            target = rewards[:,i] + self.gamma * (1 - dones) * values[:,i]
            loss = self.loss(action_values, target.detach())

            # optimize
            self.optimizer[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy[i].parameters(), 10)
            self.optimizer[i].step()
            total_loss+=loss.item()

        return total_loss/self.n_agents

    # def static_value(self, observations:np.ndarray):
    #     return self.values[np.array(observations, dtype=np.int64)]

    def learn(self, total_steps):
        step = 0
        
        while step < total_steps:
            with torch.no_grad():
                total_reward, step = self.runner.run(step, self.solver)

            if len(self.memory) < self.batch_size:
                continue
            
            loss = self.update()

            self.sync()
            
            info = {
                'Ep.Reward':total_reward[0],
                'Epsilon':self.runner.epsilon,
                'Loss':loss,
            }
            self.log_info(step, info)
            
        torch.save(self.infos, './log/'+self.config['logdir'])
        print(f"訓練結束，Gambit 求解總時間：{self.solver.total_gambit_time:.6f} 秒")

    def sync(self):
        if not self.dynamic:
            return
        
        for i in range(self.n_agents):
            target_net_weights = self.target_policy[i].state_dict()
            q_net_weights = self.policy[i].state_dict()
            for key in q_net_weights:
                target_net_weights[key] = q_net_weights[key]*self.tau + target_net_weights[key]*(1-self.tau)
            self.target_policy[i].load_state_dict(target_net_weights)

    def hard_sync(self):
        return

    def load_model(self, path=None):
        if path is None:
            path = './model/'+self.config['logdir']
        models = torch.load(path)
        for i in range(self.n_agents):
            self.policy[i].load_state_dict(models['agent'+str(i)])
            if self.dynamic:
                self.target_policy[i].load_state_dict(models['agent'+str(i)])

class DynamicSolver():
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
            self.strategy = np.zeros((self.env.n_states, self.n_agents, self.n_actions))
            self.static_values = np.zeros((self.env.n_states, self.n_agents))            
            for state in range(self.env.n_states):
                start_time = time.time()
                
                # 1. self-defined class -> gamebit form
                gamebit_form_game = DynamicSolver.from_arrays(self.env.payoff_matrix[state])
                # 2. solve nash from gambit
                result = gbt.nash.ipa_solve(gamebit_form_game).equilibria
                # 3. get probability and store
                self.strategy[state] = DynamicSolver.extract_strategy(result, self.n_agents, self.n_actions)
                # 4. calculate expected payoff for every state given nash
                self.static_values[state] = DynamicSolver.get_static_expected_payoff(self.env.payoff_matrix[state], self.strategy[state])            
                
                end_time = time.time()
                self.total_gambit_time += (end_time - start_time)
        else:
            """dynamic: slove nash based on state and Q-network"""
            self.saving = False
            

    def set_policy(self, policy, target_policy):
        self.policy = policy
        self.target_policy = target_policy

    def __call__(self, state):
        """strategy: (n_agent, n_actions)"""
        if (not self.dynamic) or self.saving:
            return np.squeeze(self.strategy[state])

        self.strategy = np.zeros((self.env.n_states, self.n_agents, self.n_actions))
        self.static_values = np.zeros((self.env.n_states, self.n_agents))
        dynamic_payoff_matrix = self.from_qvalues()
        for state in range(self.env.n_states):
            
            start_time = time.time()
            
            # 1. Q-tables -> gamebit form
            gamebit_form_game = DynamicSolver.from_arrays(dynamic_payoff_matrix[state])
            # 2. solve nash from gambit
            result = gbt.nash.ipa_solve(gamebit_form_game).equilibria
            # 3. get probability and store
            self.strategy[state] = DynamicSolver.extract_strategy(result, self.env.n_agents, self.n_actions)
            # 4. calculate expected payoff for every state given nash
            self.static_values[state] = DynamicSolver.get_static_expected_payoff(dynamic_payoff_matrix[state], self.strategy[state])
        
            end_time = time.time()
            self.total_gambit_time += (end_time - start_time)
            
        self.saving = True
        return self(state)
    
    def from_qvalues(self):
        dynamic_payoff_matrix = np.zeros(self.env.payoff_matrix.shape)
        with torch.no_grad():
            for state in range(self.env.n_states):
                feature = torch.as_tensor([state], dtype=torch.float32, device=self.device)
                for i in range(self.n_agents):
                    qvalues = self.target_policy[i](feature).cpu().numpy()
                    for a in range(self.n_actions):
                        joint_actions = [slice(None) for _ in range(self.n_agents)]
                        joint_actions[i] = a
                        idx = [state] + joint_actions + [i]
                        dynamic_payoff_matrix[tuple(idx)] = qvalues[a]
        return dynamic_payoff_matrix

    def values(self, state):
        return np.squeeze(self.static_values[state]) # (n_batches, n_agents)

    def from_arrays(payoff_matrix:np.ndarray):
        payoff_dict = {}
        n_agents = payoff_matrix.shape[-1]
        
        for i in range(n_agents):
            payoff_dict['Player '+str(i+1)] = payoff_matrix[..., i]

        return gbt.Game.from_dict(payoff_dict)

    def extract_strategy(result, n_agents, n_actions):
        nash_strategy = np.zeros((n_agents, n_actions))
        for i in range(n_agents):
            i_th_player_strategy = []
            for act in result[0]['Player '+str(i+1)]:
                i_th_player_strategy.append(act[1])
            nash_strategy[i] = np.array(i_th_player_strategy)
        return nash_strategy

    def get_static_expected_payoff(payoff_matrix:np.ndarray, nash_strategy:np.ndarray) -> np.ndarray:
        n_agents = payoff_matrix.shape[-1]
        n_actions = payoff_matrix.shape[0]
        expected_payoffs = np.zeros(n_agents)

        # 生成所有可能的行為組合，這裡使用 (0, 0, ..., 0) 到 (1, 1, ..., 1)
        all_actions = list(product([i for i in range(n_actions)], repeat=n_agents))

        # 遍歷每個行為組合
        for actions in all_actions:
            # 計算該行為組合的機率
            prob = 1.0
            for i, action in enumerate(actions):
                prob *= nash_strategy[i][action]

            # 從收益矩陣中獲取該行為組合的收益
            payoffs = payoff_matrix[actions]
            
            # 根據該行為組合的機率加權收益
            expected_payoffs += prob * payoffs

        return expected_payoffs
        



