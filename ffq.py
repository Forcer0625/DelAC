import torch
import torch.nn as nn
import numpy as np
from modules import MLPAgent
from utilis import ReplayBuffer
from runner import EGreedyRunner
from envs import NormalFormGame
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from iql import IQL
from scipy.optimize import linprog

# class IQL():
#     def __init__(self, env:NormalFormGame, config):
#         self.env = env
#         self.n_agents = 2
#         if type(self.env.n_actions) == int:
#             self.n_actions = self.env.n_actions
#         else:
#             self.n_actions = self.env.n_actions[0]
#         observation, _ = self.env.reset()

#         self.batch_size = config['batch_size']
#         self.memory_size = config['memory_size']
#         self.memory = ReplayBuffer(self.memory_size, observation[0].shape, self.n_agents)

#         self.lr = config['lr']
#         self.gamma = config['gamma']
#         self.tau = config['tau']
#         self.device = config['device']
#         self.loss = nn.MSELoss()

#         self.get_policy(observation)

#         self.get_runner(config)

#     def get_policy(self, observation):
#         self.policy = []
#         self.target_policy = []
#         self.optimizer = []
#         for i in range(2):
#             self.policy.append(MLPAgent(observation[0].reshape(-1).shape[0], self.n_actions, device=self.device))
#             self.target_policy.append(deepcopy(self.policy[i]))
#             self.optimizer.append(torch.optim.Adam(self.policy[i].parameters(), lr=self.lr)) 

#     def get_runner(self, config):
#         self.runner = EGreedyRunner(self.env, self.policy, self.memory,\
#                                     config['eps_start'], config['eps_end'], config['eps_dec'])
        
#     def learn(self, total_steps):
#         step = 0
#         agent_1_mean_reward = []
#         agent_2_mean_reward = []
        
#         while step < total_steps:
#             with torch.no_grad():
#                 total_reward, step = self.runner.run(step)
#             agent_1_mean_reward.append(total_reward[0])
#             agent_2_mean_reward.append(total_reward[1])

#             if len(self.memory) < self.batch_size:
#                 continue
            
#             loss = self.update()

#             for i in range(2):
#                 target_net_weights = self.target_policy[i].state_dict()
#                 q_net_weights = self.policy[i].state_dict()
#                 for key in q_net_weights:
#                     target_net_weights[key] = q_net_weights[key]*self.tau + target_net_weights[key]*(1-self.tau)
#                 self.target_policy[i].load_state_dict(target_net_weights)

#     def update(self, ):
#         observations, actions, rewards,\
#             dones, observations_ = self.memory.sample(self.batch_size)
        
        
#         observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)#.view(self.n_agents, *observations[0][0].shape)
#         actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
#         rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
#         dones = torch.as_tensor(dones, dtype=torch.int, device=self.device).view(-1)
        
#         observations_= torch.as_tensor(observations_, dtype=torch.float32, device=self.device)#.view(-1, *observations_[0][0].shape)

#         total_loss = 0.0
#         for i in range(self.n_agents):
#             action_values = self.policy[i](observations[:,i]).reshape(-1, self.n_actions)
#             action_values = action_values.gather(1, actions[:,i].unsqueeze(1))
#             action_values = action_values.reshape(-1)

#             # double-q
#             with torch.no_grad():
#                 estimate_action_values = self.policy[i](observations_[:,i]).reshape(-1, self.n_actions)
#                 next_action = torch.max(estimate_action_values, dim=1).indices
#                 next_action_values = self.target_policy[i](observations_[:,i]).reshape(-1, self.n_actions)
#                 next_action_values = next_action_values.gather(1, next_action.unsqueeze(1))
#                 next_action_values = next_action_values.reshape(-1)

#             # calculate loss
#             target = rewards[:,i] + self.gamma * (1 - dones) * next_action_values
#             loss = self.loss(action_values, target.detach())

#             # optimize
#             self.optimizer[i].zero_grad()
#             loss.backward()
#             self.optimizer[i].step()
#             total_loss+=loss.item()

#         return total_loss/self.n_agents

class FFQ(IQL):
    def __init__(self, env, config, friend_or_foe='foe'):
        super().__init__(env, config)
        assert self.parameter_sharing
        self.friend_or_foe = friend_or_foe

    def update(self):
        observations, actions, rewards,\
            dones, observations_ = self.memory.sample(self.batch_size)
        
        
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device).view(-1)
        observations_= torch.as_tensor(observations_, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        for i in range(2):
            # Q(s, a_i)
            q_values = self.policy[i](observations[:, i]).reshape(-1, self.n_actions)

            # gather current Q
            q_a = q_values.gather(1, actions[:, 0 if i==0 else -1].unsqueeze(1)).squeeze(1)

            # 下一狀態的雙人 Q 值 (agent_i 對 opponent)
            with torch.no_grad():
                opponent_id = 1 - i
                q_next_agent = self.target_policy[i](observations_[:, i]).reshape(-1, self.n_actions)
                q_next_opponent = self.target_policy[opponent_id](observations_[:, opponent_id]).reshape(-1, self.n_actions)

                # 建構 Q(s', a_i, a_j)
                q_joint = q_next_agent.unsqueeze(2) + q_next_opponent.unsqueeze(1)  # shape: (batch, a_i, a_j)
                v_next = self.solve_minmax(q_joint)

            target = rewards[:, 0 if i==0 else -1] + self.gamma *  v_next# * (1 - dones)
            loss = self.loss(q_a, target.detach())

            self.optimizer[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy[i].parameters(), 10)
            self.optimizer[i].step()
            total_loss += loss.item()

        return total_loss / self.n_agents

    def solve_minmax(self, Q_values):
        """
        Q_values: tensor of shape (batch_size, n_actions, n_opponent_actions)
        returns: tensor of shape (batch_size,) — the state value estimate
        """
        if self.friend_or_foe == "friend":
            # 假設對手是合作的，對 a_j 平均（也可使用特定策略分佈）
            v, _ = Q_values.max(dim=2)   # shape: [batch, a_i]
            v, _ = v.max(dim=1)          # shape: [batch]
            return v
        
        elif self.friend_or_foe == "foe":
            batch_size, a_i, a_j = Q_values.shape

            # ✅ Use closed-form if 2x2
            if a_i == 2 and a_j == 2:
                q11 = Q_values[:, 0, 0]
                q12 = Q_values[:, 0, 1]
                q21 = Q_values[:, 1, 0]
                q22 = Q_values[:, 1, 1]

                denom = q11 - q12 - q21 + q22
                numer = q22 - q21

                p = torch.zeros_like(denom)
                valid = denom != 0

                p[valid] = numer[valid] / denom[valid]
                p = torch.clamp(p, 0.0, 1.0)

                v1 = p * q11 + (1 - p) * q21
                v2 = p * q12 + (1 - p) * q22
                v = torch.min(v1, v2)
                return v

            # 🧠 fallback to LP solver
            v_values = []
            for b in range(batch_size):
                q = Q_values[b].detach().cpu().numpy()
                c = np.zeros(a_i + 1)
                c[-1] = -1

                A = []
                b_ub = []
                for j in range(a_j):
                    row = [-q[i][j] for i in range(a_i)]
                    row.append(1.0)
                    A.append(row)
                    b_ub.append(0)

                A_eq = [[1.0]*a_i + [0.0]]
                b_eq = [1.0]
                bounds = [(0, 1)] * a_i + [(None, None)]

                try:
                    res = linprog(c, A_ub=A, b_ub=b_ub,
                                A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                                method='highs-ds')
                    if res.success:
                        v_values.append(res.x[-1])
                    else:
                        print(f"[WARNING] linprog failed at batch {b}")
                        v_values.append(0.0)
                except Exception as e:
                    print(f"[ERROR] linprog exception: {e}")
                    v_values.append(0.0)

            return torch.tensor(v_values, dtype=torch.float32, device=self.device)
        # try:
        #     v_values = []

        #     for b in range(self.batch_size):
        #         q = Q_values[b].detach().cpu().numpy()

        #         # Foe: Minimax: max_pi min_opponent ∑ Q(s, a_i, a_j)
        #         c = np.zeros(self.n_actions + 1)
        #         c[-1] = -1  # max v → min -v

        #         A = []
        #         b_ub = []
        #         for j in range(q.shape[1]):  # for each opponent action
        #             row = [-q[i][j] for i in range(self.n_actions)]
        #             row.append(1.0)  # for v
        #             A.append(row)
        #             b_ub.append(0)

        #         A_eq = [[1.0]*self.n_actions + [0.0]]
        #         b_eq = [1.0]
        #         bounds = [(0, 1)] * self.n_actions + [(None, None)]

        #         res = linprog(c, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        #         if not res.success:
        #             raise ValueError("Linear program failed to solve minimax.")

        #         v = res.x[-1]

        #         v_values.append(v)

        #     return torch.tensor(v_values, dtype=torch.float32, device=self.device)
        # except:
        #     return torch.zeros(Q_values.shape[0], device=self.device)