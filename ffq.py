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
            # ✅ Use closed-form if 2x2
            if self.n_actions == 2:
                a = Q_values[:, 0, 0]
                b = Q_values[:, 0, 1]
                c = Q_values[:, 1, 0]
                d = Q_values[:, 1, 1]

                denom = a - b - c + d
                v = torch.zeros_like(denom)

                valid = ~torch.isclose(denom, v, atol=1e-3)#denom != 0
                v[valid] = (a[valid] * d[valid] - b[valid] * c[valid]) / denom[valid]

                # 🛡️ fallback: min-max (worst-case opponent)
                fallback = Q_values.max(dim=1).values.min(dim=1).values
                v[~valid] = fallback[~valid]

                return v

            # 🧠 fallback to LP solver
            v_values = []
            for b in range(self.batch_size):
                q = Q_values[b].detach().cpu().numpy()
                c = np.zeros(self.n_actions + 1)
                c[-1] = -1

                A = []
                b_ub = []
                for j in range(self.n_actions):
                    row = [-q[i][j] for i in range(self.n_actions)]
                    row.append(1.0)
                    A.append(row)
                    b_ub.append(0)

                A_eq = [[1.0]*self.n_actions + [0.0]]
                b_eq = [1.0]
                bounds = [(0, 1)] * self.n_actions + [(None, None)]

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