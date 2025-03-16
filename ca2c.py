import torch
import torch.nn as nn
import time
import json
from copy import deepcopy
from modules import Actor, CentralisedCritic
from runner import CentralisedOnPolicyRunner
from ia2c import IA2C
from torch.utils.tensorboard import SummaryWriter

class CA2C(IA2C):
    def get_policy(self):
        self.v_loss = nn.MSELoss()
        self.critic = CentralisedCritic(input_dim=self.input_dim, action_dim=self.n_agents, value_dim=2, device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr) 

        self.actors = []
        self.actor_optim = []
        if self.parameter_sharing:
            team_1_policy = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            team_2_policy = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            team_1_optim = torch.optim.Adam(team_1_policy.parameters(), lr=self.lr)
            team_2_optim = torch.optim.Adam(team_2_policy.parameters(), lr=self.lr)
            for i in range(self.n_agents):
                self.actors.append(team_1_policy if i < self.n_agents//2 else team_2_policy)
                self.actor_optim.append(team_1_optim if i < self.n_agents//2 else team_2_optim)
        else:
            for i in range(self.n_agents):
                self.actors.append(Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device))
                self.actor_optim.append(torch.optim.Adam(self.actors[i].parameters(), lr=self.lr))
    
    def linear_lr_decay(self, step, total_steps):
        lr = self.lr - (self.lr * (step / float(total_steps)))
        
        for actor_optim in self.actor_optim:
            for param_group in actor_optim.param_groups:
                param_group['lr'] = lr

        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr
                
    def learn(self, total_steps):
        self.save_config()
        steps = 0
        runtime_iterations = 0
        total_episodes = 0
        t_start = time.time()
        
        while steps < total_steps:
            with torch.no_grad():
                mb_obs, mb_actions, mb_values, mb_returns, episodes = self.runner.run(self.actors, self.critic)
                mb_advs = []
                for i in range(self.n_agents):
                    mb_advs.append(mb_returns[i] - mb_values[i])
                    mb_advs[i] = (mb_advs[i] - mb_advs[i].mean()) / (mb_advs[i].std() + 1e-8)
            
            total_pg_loss = 0.0
            for i in range(self.n_agents):
                pg_loss, entropy = self.update_actor(mb_obs[i], mb_actions[i], mb_advs[i], i)
                total_pg_loss += pg_loss
            
            mb_actions = mb_actions.transpose()
            mb_returns = mb_returns[self.n_agents//2-1:self.n_agents//2+1].transpose()
            v_loss = self.update_critic(mb_obs[i], mb_actions, mb_returns)

            steps += self.runner.n_env * self.batch_size
            
            self.linear_lr_decay(steps, total_steps)
            self.linear_ent_coef_decay(steps, total_steps)

            mean_return, std_return, mean_len = self.runner.get_performance()
            info = {
                'Team1-Ep.Reward':mean_return[ 0],
                'Team2-Ep.Reward':mean_return[-1],
                'Team1-Std.Reward':std_return[ 0],
                'Team2-Std.Reward':std_return[-1],
                'Loss.Actor':total_pg_loss,
                'Loss.Critic':v_loss,
                'Entropy': entropy,
            }
            self.log_info(steps, info)
            
            total_episodes += episodes
            runtime_iterations += 1
            if runtime_iterations % self.print_every == 0:
                n_sec = time.time() - t_start
                fps = int(runtime_iterations*self.runner.n_env*self.batch_size / n_sec)

                print("[{:5d} / {:5d}]".format(steps, total_steps))
                print("----------------------------------")
                print("Elapsed time = {:.2f} sec".format(n_sec))
                print("FPS          = {:d}".format(fps))
                print("actor loss   = {:.6f}".format(total_pg_loss))
                print("critic loss  = {:.6f}".format(v_loss))
                print("entropy      = {:.6f}".format(entropy))
                print("Team1 mean return  = {:.6f}".format(mean_return[ 0]))
                print("Team2 mean return  = {:.6f}".format(mean_return[-1]))
                print("Team1 std return  = {:.6f}".format(std_return[ 0]))
                print("Team2 std return  = {:.6f}".format(std_return[-1]))
                print("mean length  = {:.2f}".format(mean_len))
                print("total episode= {:d}".format(total_episodes))
                print("iterations   = {:d}".format(runtime_iterations))
                print()

        self.runner.close()
        self.save_model()
        torch.save(self.infos, './log/'+self.config['logdir'])
        print("----Training End----")
    
    def update_actor(self, mb_observations, mb_actions, mb_advs, agent_idx):
        mb_observations= torch.from_numpy(mb_observations).to(self.device)
        mb_actions     = torch.from_numpy(mb_actions).to(self.device)
        mb_advs        = torch.from_numpy(mb_advs).to(self.device)

        a_logps, ents = self.actors[agent_idx].evaluate(mb_observations, mb_actions)
        
        #Actor Loss
        pg_loss = -(a_logps * mb_advs).mean() - self.ent_coef * ents.mean()

        #Train actor
        self.actor_optim[agent_idx].zero_grad()
        pg_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.grad_norm)
        self.actor_optim[agent_idx].step()

        return pg_loss.item(), ents.mean().item()
    
    def update_critic(self, mb_observations, mb_actions, mb_returns):
        mb_observations = torch.from_numpy(mb_observations).to(self.device)
        mb_actions      = torch.from_numpy(mb_actions).to(self.device)
        mb_returns      = torch.from_numpy(mb_returns).to(self.device)

        values = self.critic(mb_observations, mb_actions) # (batch_size, 2)

        #Critic Loss
        v_loss = self.v_loss(values, mb_returns)

        #Train critic
        self.critic_optim.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
        self.critic_optim.step()
        
        return v_loss.item()