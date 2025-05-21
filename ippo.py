from ia2c import IA2C
import numpy as np
import torch
import torch.nn as nn
import time
from modules import ActorCritic

class IPPO(IA2C):
    def __init__(self, runner, config):
        super().__init__(runner, config)
        self.clip_param = config['clip_param']
        self.epochs = config['epochs']
        self.sample_mb_size = config['sample_mb_size']
        self.mb_size = config['batch_size']*config['n_env']
        self.sample_n_mb = self.mb_size // self.sample_mb_size
        self.rand_idx = np.arange(self.mb_size)

    def update(self, mb_observations, mb_actions,  mb_old_values, mb_advs, mb_returns, mb_old_a_logps, target_policy:ActorCritic):
        mb_observations= torch.from_numpy(mb_observations).to(self.device)
        mb_actions     = torch.from_numpy(mb_actions).to(self.device)
        mb_old_values  = torch.from_numpy(mb_old_values).to(self.device)
        mb_advs        = torch.from_numpy(mb_advs).to(self.device)
        mb_returns     = torch.from_numpy(mb_returns).to(self.device)
        mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)

        for _ in range(self.epochs):
            np.random.shuffle(self.rand_idx)

            for j in range(self.sample_n_mb):
                sample_idx         = self.rand_idx[j*self.sample_mb_size : (j+1)*self.sample_mb_size]
                sample_observations= mb_observations[sample_idx]
                sample_actions     = mb_actions[sample_idx]
                sample_old_values  = mb_old_values[sample_idx]
                sample_advs        = mb_advs[sample_idx]
                sample_returns     = mb_returns[sample_idx]
                sample_old_a_logps = mb_old_a_logps[sample_idx]

                sample_a_logps, sample_ents = target_policy.evaluate(sample_observations, sample_actions)
                sample_values = target_policy.critic(sample_observations, sample_actions)

                #PPO loss
                v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_param, self.clip_param)
                v_loss1     = (sample_returns - sample_values).pow(2)
                v_loss2     = (sample_returns - v_pred_clip).pow(2)
                v_loss      = torch.max(v_loss1, v_loss2).mean()

                ratio = torch.exp(sample_a_logps - sample_old_a_logps)
                clip_adv = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * sample_advs
                pg_loss = -torch.min(ratio * sample_advs, clip_adv).mean() - self.ent_coef*sample_ents.mean()

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

        return pg_loss.item(), v_loss.item(), sample_ents.mean().item()
    
    def learn(self, total_steps):
        self.save_config()
        steps = 0
        runtime_iterations = 0
        total_episodes = 0
        t_start = time.time()
        
        while steps < total_steps:
            with torch.no_grad():
                mb_obs, mb_actions, mb_a_logps, mb_values, mb_returns, episodes = self.runner.run(self.actor_critic)
                mb_advs = []
                for i in range(self.n_agents):
                    mb_advs.append(mb_returns[i] - mb_values[i])
                    mb_advs[i] = (mb_advs[i] - mb_advs[i].mean()) / (mb_advs[i].std() + 1e-8)
            
            total_pg_loss = 0.0
            total_v_loss = 0.0
            for i in range(self.n_agents):
                pg_loss, v_loss, entropy = self.update(mb_obs[i], mb_actions[i], mb_values[i], mb_advs[i], mb_returns[i], mb_a_logps[i], self.actor_critic[i])
                total_pg_loss += pg_loss
                total_v_loss += v_loss

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