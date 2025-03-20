import torch
import torch.nn as nn
import time
import json
from copy import deepcopy
from modules import ActorCritic
from runner import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

class IA2C():
    def __init__(self, runner: OnPolicyRunner, config):
        self.n_agents = runner.n_agents
        self.env = runner.env

        self.lr = config['lr']
        self.ent_coef = config['ent_coef']
        self.grad_norm = config['grad_norm']
        self.device  = config['device']
        self.parameter_sharing = config['use-parameter-sharing']
        
        self.batch_size = config['batch_size']
        self.mb_size = config['batch_size']*config['n_env']

        self.input_dim = 1
        self.action_dim = 2

        self.runner = runner
        
        self.logger = SummaryWriter('./runs/'+config['logdir'])
        self.infos = []
        self.print_every = config['print_every']
        self.config = config

        self.get_policy()
    
    def get_policy(self):
        self.actor_critic = [] 
        if self.parameter_sharing:
            team_1_policy = ActorCritic(input_dim=self.input_dim, action_dim=self.action_dim, lr=self.lr, device=self.device)
            team_2_policy = ActorCritic(input_dim=self.input_dim, action_dim=self.action_dim, lr=self.lr, device=self.device)
            for i in range(self.n_agents):
                self.actor_critic.append(team_1_policy if i < self.n_agents//2 else team_2_policy)
        else:
            for _ in range(self.n_agents):
                self.actor_critic.append(ActorCritic(input_dim=self.input_dim, action_dim=self.action_dim, lr=self.lr, device=self.device))
    
    def linear_lr_decay(self, step, total_steps):
        lr = self.lr - (self.lr * (step / float(total_steps)))
        
        for policy in self.actor_critic:
            for param_group in policy.actor_optim.param_groups:
                param_group['lr'] = lr

            for param_group in policy.critic_optim.param_groups:
                param_group['lr'] = lr
    
    def linear_ent_coef_decay(self, step, total_steps):
        self.ent_coef = self.config['ent_coef'] - (self.config['ent_coef'] * (step / float(total_steps)))

    def log_info(self, step, info:dict):
        for key in info.keys():
            if key != 'Steps':
                self.logger.add_scalar('Train/'+key, info[key], step)
        self.infos.append(info)
                
    def learn(self, total_steps):
        self.save_config()
        steps = 0
        runtime_iterations = 0
        total_episodes = 0
        t_start = time.time()
        
        while steps < total_steps:
            with torch.no_grad():
                mb_obs, mb_actions, mb_values, mb_returns, episodes = self.runner.run(self.actor_critic)
                mb_advs = []
                for i in range(self.n_agents):
                    mb_advs.append(mb_returns[i] - mb_values[i])
                    mb_advs[i] = (mb_advs[i] - mb_advs[i].mean()) / (mb_advs[i].std() + 1e-8)
            
            total_pg_loss = 0.0
            total_v_loss = 0.0
            for i in range(self.n_agents):
                pg_loss, v_loss, entropy = self.update(mb_obs[i], mb_actions[i], mb_advs[i], mb_returns[i], self.actor_critic[i])
                total_pg_loss += pg_loss
                total_v_loss += v_loss

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
                'Loss.Critic':total_v_loss,
                'Entropy': entropy,
                'Step':steps
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
                print("critic loss  = {:.6f}".format(total_v_loss))
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
    
    def update(self, mb_observations, mb_actions, mb_advs, mb_returns, target_policy:ActorCritic):
        mb_observations= torch.from_numpy(mb_observations).to(self.device)
        mb_actions     = torch.from_numpy(mb_actions).to(self.device)
        mb_advs        = torch.from_numpy(mb_advs).to(self.device)
        mb_returns     = torch.from_numpy(mb_returns).to(self.device)

        a_logps, ents = target_policy.evaluate(mb_observations, mb_actions)
        values = target_policy.critic(mb_observations, mb_actions)
        
        #Actor Loss
        pg_loss = -(a_logps * mb_advs).mean() - self.ent_coef * ents.mean()

        # Criric Loss
        v_loss = (mb_returns - values).pow(2).mean()

        #Train actor
        target_policy.actor_optim.zero_grad()
        pg_loss.backward()
        nn.utils.clip_grad_norm_(target_policy.actor.parameters(), self.grad_norm)
        target_policy.actor_optim.step()

        #Train critic
        target_policy.critic_optim.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(target_policy.critic.parameters(), self.grad_norm)
        target_policy.critic_optim.step()

        return pg_loss.item(), v_loss.item(), ents.mean().item()
    
    def save_config(self):
        config = deepcopy(self.config)
        config['device'] = self.config['device'].type
        with open('./runs/'+self.config['logdir']+'/config.json', 'w') as f:
            json.dump(config, f)
    
    def save_model(self, path=None):
        pass
    
    def load_model(self, path=None):
        pass