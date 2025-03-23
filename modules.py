import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Orthogonal initialization function
def init(module:nn.Module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
class RNNAgent(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=64):
        super(RNNAgent, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
class MLPAgent(nn.Module):
    def __init__(self, observation_dim, n_actions, hidden_dim=128, device=torch.device('cpu')):
        super(MLPAgent, self).__init__()

        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, n_actions)
        self.device = device
        self.to(device)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.q(x)
    
class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim=64, device=torch.device('cpu')):
        super(QMixer, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.device = device

        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.n_agents * self.hidden_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim))

        self.hyper_b1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, 1))

        self.to(device)

    def forward(self, q_values, states):
        #states = states.reshape(-1, self.state_dim)
        #q_values = q_values.reshape(-1, 1, self.n_agents)
        
        w_1 = torch.abs(self.hyper_w1(states))
        w_1 = w_1.view(-1, self.n_agents, self.hidden_dim)
        b_1 = self.hyper_b1(states)
        b_1 = b_1.view(-1, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w_1) + b_1)

        w_2 = torch.abs(self.hyper_w2(states))
        w_2 = w_2.view(-1, self.hidden_dim, 1)
        b_2 = self.hyper_b2(states)
        b_2 = b_2.view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w_2 ) + b_2
        q_tot = q_tot.view(-1, 1)

        return q_tot
    
class NWQMixer(QMixer):
    def __init__(self, state_dim, n_agents, team, hidden_dim=64, device=torch.device('cpu')):
        super().__init__(state_dim, n_agents, hidden_dim, device)
        self.team = team

    def forward(self, q_values, states):
        w_1 = torch.abs(self.hyper_w1(states))
        w_1 = w_1.view(-1, self.n_agents, self.hidden_dim)
        # add negative weight to opponents
        if self.team == 0:
            w_1[:,self.n_agents//2:,:] = -w_1[:,self.n_agents//2:,:]
        else:
            w_1[:,:self.n_agents//2,:] = -w_1[:,:self.n_agents//2,:]
        b_1 = self.hyper_b1(states)
        b_1 = b_1.view(-1, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w_1) + b_1)

        w_2 = torch.abs(self.hyper_w2(states))
        w_2 = w_2.view(-1, self.hidden_dim, 1)
        b_2 = self.hyper_b2(states)
        b_2 = b_2.view(-1, 1, 1)

        q_tot = torch.bmm(hidden, w_2 ) + b_2
        q_tot = q_tot.view(-1, 1)

        return q_tot

#Normal distribution module with fixed mean and std.
class FixedNormal(torch.distributions.Normal):
	# Log-probability
	def log_prob(self, actions):
		return super().log_prob(actions).sum(-1)

	# Entropy
	def entropy(self):
		return super().entropy().sum(-1)

	# Mode
	def mode(self):
		return self.mean

#Diagonal Gaussian distribution
class DiagGaussian(nn.Module):
    # Constructor
    def __init__(self, inp_dim, out_dim, std=0.5):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0)
        )
        self.fc_mean = init_(nn.Linear(inp_dim, out_dim))
        self.log_std = nn.Parameter(torch.log(torch.full((out_dim,), std)))
        #self.std = torch.full((out_dim,), std) use learnable std

    # Forward
    def forward(self, x):
        mean = self.fc_mean(x)
        std = self.log_std.exp()
        return FixedNormal(mean, std.to(x.device))

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, std=0.5, continous_action=False):
        super(Actor, self).__init__()
        init_ = lambda m: init(
			m,
			nn.init.orthogonal_,
			lambda x: nn.init.constant_(x, 0),
			nn.init.calculate_gain('relu')
		)
        self.continous_action = continous_action
        if self.continous_action:
            self.model = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)),
                nn.Tanh(),
                init_(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
            )
            self.dist = DiagGaussian(hidden_dim, action_dim, std=std)
        else:
            self.model = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_dim)),
                nn.Tanh(),
                init_(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                init_(nn.Linear(hidden_dim, action_dim)),
                nn.Softmax(-1),
            )
            self.dist = torch.distributions.Categorical

    def forward(self, observation, deterministic=False):
        feature = self.model(observation)
        dist    = self.dist(feature)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist.log_prob(action)
    
    def select_action(self, observation, deterministic=False):
        feature = self.model(observation)
        dist    = self.dist(feature)
        
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action
    
    def evaluate(self, observation, action):
        feature = self.model(observation)
        dist    = self.dist(feature)
        return dist.log_prob(action), dist.entropy()
   
class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, value_dim=1, continous_action=False):
        super(Critic, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu')
        )
        self.continous_action = continous_action
        if not continous_action:
            action_dim = 1
        self.model = nn.Sequential(
            init_(nn.Linear(input_dim+action_dim, hidden_dim)),
            nn.Tanh(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_(nn.Linear(hidden_dim, value_dim))
        )

    def forward(self, observation, action):
        if self.continous_action:    
            return self.model(torch.cat((observation, action), 1))[:, 0]
        return self.model(torch.cat((observation, action.unsqueeze(1)), 1))[:, 0]

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, lr=3e-4, hidden_dim=128, eps=1e-8, device=torch.device('cpu')):
        super(ActorCritic, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.actor = Actor(input_dim, action_dim, hidden_dim)
        self.critic = Critic(input_dim, action_dim, hidden_dim)
        self.device = device
        self.to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=eps)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr*100.0, eps=eps)

    def forward(self, observation, action):
        '''Returns action probs, log probs and value '''        
        sample_action, a_logs = self.actor(observation)
        value = self.critic(observation, action)
        return sample_action, a_logs, value

    def select_action(self, observation, deterministic=False):
        '''Returns action only'''
        return self.actor.select_action(observation, deterministic)
    
    def evaluate(self, observation, action):
        '''Return log probs & entropy'''
        return self.actor.evaluate(observation, action)
    
class CentralisedCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, value_dim=2, device=torch.device('cpu')):
        super(CentralisedCritic, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu')
        )
        self.actoin_dim = action_dim
        self.model = nn.Sequential(
            init_(nn.Linear(input_dim+2, hidden_dim)),
            nn.Tanh(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            init_(nn.Linear(hidden_dim, value_dim))
        )
        self.device = device
        self.to(device)

    def forward(self, observation, action):
        if action.dim() == 1:
            action = action.unsqueeze(0)  # 轉為 (1, 4)

        # 分別計算前 2 個元素與後 2 個元素的 1.0 個數
        team1_count = torch.sum(action[:, :self.actoin_dim//2], dim=1)
        team2_count = torch.sum(action[:, self.actoin_dim//2:], dim=1)

        # 合併結果，維持 batch 維度
        joint_action_indicator = torch.stack([team1_count, team2_count], dim=1)

        # 若原本無 batch 維度，則壓縮回 (2,)
        action = joint_action_indicator.squeeze(0) if action.shape[0] == 1 else joint_action_indicator
        feature = torch.cat((observation, action), 0 if observation.dim() == 1 else 1)
        x = self.model(feature)
        return x