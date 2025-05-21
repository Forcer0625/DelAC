from ca2c import CA2C
import numpy as np
import torch
import torch.nn as nn
import time

class MAPPO(CA2C):
    def __init__(self, runner, config):
        super().__init__(runner, config)
        self.clip_param = config['clip_param']
        self.epochs = config['epochs']
        self.sample_mb_size = config['sample_mb_size']
        self.mb_size = config['batch_size']*config['n_env']
        self.sample_n_mb = self.mb_size // self.sample_mb_size
        self.rand_idx = np.arange(self.mb_size)

    def learn(self, total_steps):
        self.save_config()
        steps = 0
        runtime_iterations = 0
        total_episodes = 0
        t_start = time.time()
        
        while steps < total_steps:
            with torch.no_grad():
                mb_obs, mb_actions, mb_a_logps, mb_values, mb_returns, episodes = self.runner.run(self.actors, self.critic)
                mb_advs = []
                for i in range(self.n_agents):
                    mb_advs.append(mb_returns[i] - mb_values[i])
                    mb_advs[i] = (mb_advs[i] - mb_advs[i].mean()) / (mb_advs[i].std() + 1e-8)
            
            total_pg_loss = 0.0
            for i in range(self.n_agents):
                pg_loss, entropy = self.update_actor(mb_obs[i], mb_actions[i], mb_advs[i], mb_a_logps[i], i)
                total_pg_loss += pg_loss
            
            mb_actions = mb_actions.transpose()
            mb_returns = mb_returns[self.n_agents//2-1:self.n_agents//2+1].transpose()
            mb_values  = mb_values[self.n_agents//2-1:self.n_agents//2+1].transpose()
            v_loss = self.update_critic(mb_obs[i], mb_actions, mb_values, mb_returns)

            steps += self.batch_size
            
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
    
    def update_actor(self, mb_observations, mb_actions, mb_advs, mb_old_a_logps, agent_idx):
        mb_observations= torch.from_numpy(mb_observations).to(self.device)
        mb_actions     = torch.from_numpy(mb_actions).to(self.device)
        mb_advs        = torch.from_numpy(mb_advs).to(self.device)
        mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)

        a_logps, ents = self.actors[agent_idx].evaluate(mb_observations, mb_actions)
        
        #Actor Loss
        ratio = torch.exp(a_logps - mb_old_a_logps)
        clip_adv = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * mb_advs
        pg_loss = -torch.min(ratio * mb_advs, clip_adv).mean() - self.ent_coef*ents.mean()

        #Train actor
        self.actor_optim[agent_idx].zero_grad()
        pg_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.grad_norm)
        self.actor_optim[agent_idx].step()

        return pg_loss.item(), ents.mean().item()
    
    def update_critic(self, mb_observations, mb_actions, mb_old_values, mb_returns):
        mb_observations = torch.from_numpy(mb_observations).to(self.device)
        mb_actions      = torch.from_numpy(mb_actions).to(self.device)
        mb_old_values   = torch.from_numpy(mb_old_values).to(self.device)
        mb_returns      = torch.from_numpy(mb_returns).to(self.device)

        values = self.critic(mb_observations, mb_actions) # (batch_size, 2)

        #Critic Loss
        v_pred_clip = mb_old_values + torch.clamp(values - mb_old_values, -self.clip_param, self.clip_param)
        v_loss1     = (mb_returns - values).pow(2)
        v_loss2     = (mb_returns - v_pred_clip).pow(2)
        v_loss      = torch.max(v_loss1, v_loss2).mean()        

        #Train critic
        self.critic_optim.zero_grad()
        v_loss.backward()
        self.critic_optim.step()
        
        return v_loss.item()