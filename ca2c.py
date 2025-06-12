import torch
import torch.nn as nn
import time
import json
import numpy as np
from copy import deepcopy
from modules import Actor, CentralisedCritic
from runner import CentralisedOnPolicyRunner
from ia2c import IA2C
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import csv
from team_feasibility import feasibility_run
from judger import NashEquilibriumJudger
from iql import DynamicSolver
from envs import TwoTeamSymmetricGame

class CA2C(IA2C):
    def get_policy(self):
        self.v_loss = nn.MSELoss()
        self.critic = CentralisedCritic(input_dim=self.input_dim, action_dim=self.n_agents, value_dim=2, device=self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr*100.0) 

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

        # for param_group in self.critic_optim.param_groups:
        #     param_group['lr'] = lr
                
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
            mb_returns = mb_returns[self.n_agents//2-1:self.n_agents//2].transpose()
            v_loss = self.update_critic(mb_obs[i], mb_actions, mb_returns)

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
        #nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
        self.critic_optim.step()
        
        return v_loss.item()
    
    def make_payoff_matrix(self):
        shape = [2 for _ in range(self.n_agents)]
        shape.append(self.n_agents)
        payoff_matrix = np.zeros(shape=shape)
        with torch.no_grad():
            all_actions = list(product([0, 1], repeat=self.n_agents))
            for joint_action in all_actions:
                obs = torch.as_tensor([0]).float().to(self.device)
                actions = torch.as_tensor(joint_action).float().to(self.device)
                team_payoffs = self.critic(obs, actions).cpu().numpy()

                payoff_matrix[joint_action][:self.n_agents//2] = round(team_payoffs[0], 1)
                payoff_matrix[joint_action][self.n_agents//2:] = round(team_payoffs[1], 1)

        return payoff_matrix
    
    def save_game(self, game_type):
        if self.n_agents == 4 and self.action_dim == 2:
            logdir = self.config['logdir']
            payoff_matrix = self.make_payoff_matrix()
            game = game_type(self.n_agents)
            game.set_payoff_matrix(payoff_matrix)
            with open('./runs/'+logdir+'/critic_learned.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for player4_action in [0, 1]:
                    writer.writerow(['Player 4: '+str(player4_action)])
                    for player3_action in [0, 1]:
                        writer.writerow(['Player 3: '+str(player3_action), 'Player 2: 0', 'Player 2: 1'])
                        for player1_action in [0, 1]:
                            write_data = ['Player 1: '+str(player1_action)]
                            for player2_acton in [0, 1]:
                                joint_action = (player1_action, player2_acton, player3_action, player4_action)
                                pay_off_str = ', '.join(str(round(x, 1)) for x in game.payoff_matrix[joint_action])
                                write_data.append(pay_off_str)
                            writer.writerow(write_data)
                        writer.writerow([])
                    writer.writerow([])

                # feasibility
                strategy, _, _ = feasibility_run(game)
                strategy = strategy.reshape(self.n_agents, 2)
                expected_payoff = NashEquilibriumJudger.get_payoff(strategy, game.payoff_matrix)
                player_strategy_str = []
                for i in range(self.n_agents):
                    player_strategy_str.append('[' + ', '.join(str(round(x, 4)) for x in strategy[i]) + ']')
                writer.writerow(player_strategy_str)

                writer.writerow([])
                writer.writerow(['feasibility'])
                writer.writerow(['Player '+str(i+1) for i in range(self.n_agents)])
                player_expected_payoff = []
                for i in range(self.n_agents):
                    player_expected_payoff.append(str(round(expected_payoff[i], 4)))
                writer.writerow(player_expected_payoff)

class CFAC(CA2C):
    def __init__(self, runner, config):
        if config['n_states'] != 1:
            raise Exception('CFAC can only run on 1-state game')
        super().__init__(runner, config)
        
    def learn(self, total_steps):
        self.save_config()
        steps = 0
        runtime_iterations = 0
        total_episodes = 0
        t_start = time.time()
        
        while steps < total_steps:
            with torch.no_grad():
                mb_obs, mb_actions, mb_values, mb_returns, episodes = self.runner.run(self.actors, self.critic)

            steps +=  self.batch_size
            total_episodes += episodes
            runtime_iterations += 1
            
            mb_actions = mb_actions.transpose()
            mb_returns = mb_returns[self.n_agents//2-1:self.n_agents//2].transpose()
            v_loss = self.update_critic(mb_obs[0], mb_actions, mb_returns)

            total_pg_loss = entropy = 0.0
            payoff_matrix = self.make_payoff_matrix()
            game = TwoTeamSymmetricGame(self.n_agents)
            game.set_payoff_matrix(payoff_matrix)
            strategy, _, _ = feasibility_run(game, self.n_agents)
            valid = True
            if np.any(strategy < 0.0):
                valid = False
            else:
                strategy = strategy.reshape((self.n_agents, self.action_dim))
                for i in range(self.n_agents):
                    pg_loss, entropy = self.update_actor(mb_obs[i], strategy[i], i)
                    total_pg_loss += pg_loss

            mean_return, std_return, mean_len = self.runner.get_performance()
            info = {
                'Team1-Ep.Reward':mean_return[ 0],
                'Team2-Ep.Reward':mean_return[-1],
                'Team1-Std.Reward':std_return[ 0],
                'Team2-Std.Reward':std_return[-1],
                'Loss.Critic':v_loss,
                'Step':steps
            }
            if valid:
                info['Loss.Actor'] = total_pg_loss
                info['Entropy'] = entropy
            self.log_info(steps, info)
            
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

    def update_actor(self, mb_observations, target_dist, agent_idx):
        mb_observations= torch.from_numpy(mb_observations).to(self.device)
        target_dist = torch.from_numpy(target_dist).to(self.device)
        target_dist = target_dist.unsqueeze(0).expand(mb_observations.shape[0], -1)

        a_probs = self.actors[agent_idx].model(mb_observations)
        kl_loss = nn.functional.kl_div(a_probs.log(), target_dist, reduction="batchmean")
        with torch.no_grad():
            dist = self.actors[agent_idx].dist(a_probs)
            ents = dist.entropy()

        #Train actor
        self.actor_optim[agent_idx].zero_grad()
        kl_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.grad_norm)
        self.actor_optim[agent_idx].step()

        return kl_loss.item(), ents.mean().item()
        
class CFAC2(CFAC):
    def get_policy(self):
        self.v_loss = nn.MSELoss()

        self.actors = []
        self.actor_optim = []
        self.critics= []
        self.critic_optim = []
        if self.parameter_sharing:
            # team_1_policy = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            # team_2_policy = Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device)
            # team_1_optim = torch.optim.Adam(team_1_policy.parameters(), lr=self.lr)
            # team_2_optim = torch.optim.Adam(team_2_policy.parameters(), lr=self.lr)
            # for i in range(self.n_agents):
            #     self.actors.append(team_1_policy if i < self.n_agents//2 else team_2_policy)
            #     self.actor_optim.append(team_1_optim if i < self.n_agents//2 else team_2_optim)
            pass
        else:
            for i in range(self.n_agents):
                self.actors.append(Actor(input_dim=self.input_dim, action_dim=self.action_dim).to(self.device))
                self.actor_optim.append(torch.optim.Adam(self.actors[i].parameters(), lr=self.lr))
                self.critics.append(CentralisedCritic(input_dim=self.input_dim, action_dim=self.n_agents, value_dim=1, device=self.device))
                self.critic_optim.append(torch.optim.Adam(self.critics[i].parameters(), lr=self.lr*100.0) )

    def learn(self, total_steps):
        self.save_config()
        steps = 0
        runtime_iterations = 0
        total_episodes = 0
        t_start = time.time()
        
        while steps < total_steps:
            with torch.no_grad():
                mb_obs, mb_actions, mb_values, mb_returns, episodes = self.runner.run(self.actors, self.critics)

            steps +=  self.batch_size
            total_episodes += episodes
            runtime_iterations += 1
            
            mb_actions = mb_actions.transpose() # (n_agents, batch_size) -> (batch_size, n_agents)
            mb_returns = mb_returns.transpose()
            v_loss = 0
            for i in range(self.n_agents):
                v_loss += self.update_critic(mb_obs[0], mb_actions, mb_returns, i)
            v_loss = v_loss / self.n_agents

            # total_pg_loss = entropy = 0.0
            # payoff_matrix = self.make_payoff_matrix()
            # game = TwoTeamSymmetricGame(self.n_agents)
            # game.set_payoff_matrix(payoff_matrix)
            # strategy, _, _ = feasibility_run(game, self.n_agents)
            # valid = True
            # if np.any(strategy < 0.0):
            #     valid = False
            # else:
            #     strategy = strategy.reshape((self.n_agents, self.action_dim))
            #     for i in range(self.n_agents):
            #         pg_loss, entropy = self.update_actor(mb_obs[i], strategy[i], i)
            #         total_pg_loss += pg_loss

            mean_return, std_return, mean_len = self.runner.get_performance()
            info = {
                'Team1-Ep.Reward':mean_return[ 0],
                'Team2-Ep.Reward':mean_return[-1],
                'Team1-Std.Reward':std_return[ 0],
                'Team2-Std.Reward':std_return[-1],
                'Loss.Critic':v_loss,
                'Step':steps
            }
            # if valid:
            #     info['Loss.Actor'] = total_pg_loss
            #     info['Entropy'] = entropy
            self.log_info(steps, info)
            
            if runtime_iterations % self.print_every == 0:
                n_sec = time.time() - t_start
                fps = int(runtime_iterations*self.runner.n_env*self.batch_size / n_sec)

                print("[{:5d} / {:5d}]".format(steps, total_steps))
                print("----------------------------------")
                print("Elapsed time = {:.2f} sec".format(n_sec))
                print("FPS          = {:d}".format(fps))
                #print("actor loss   = {:.6f}".format(total_pg_loss))
                print("critic loss  = {:.6f}".format(v_loss))
                #print("entropy      = {:.6f}".format(entropy))
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

    def update_critic(self, mb_observations, mb_actions, mb_returns, agent_idx):
        mb_observations = torch.from_numpy(mb_observations).to(self.device)
        mb_actions      = torch.from_numpy(mb_actions).to(self.device)
        mb_returns      = torch.from_numpy(mb_returns[:, agent_idx]).to(self.device).unsqueeze(1)

        values = self.critics[agent_idx](mb_observations, mb_actions) # (batch_size, 1)

        #Critic Loss
        v_loss = self.v_loss(values, mb_returns)

        #Train critic
        self.critic_optim[agent_idx].zero_grad()
        v_loss.backward()
        self.critic_optim[agent_idx].step()
        
        return v_loss.item()
    
    def make_payoff_matrix(self):
        shape = [2 for _ in range(self.n_agents)]
        shape.append(self.n_agents)
        payoff_matrix = np.zeros(shape=shape)
        with torch.no_grad():
            all_actions = list(product([0, 1], repeat=self.n_agents))
            for joint_action in all_actions:
                obs = torch.as_tensor([0]).float().to(self.device)
                actions = torch.as_tensor(joint_action).float().to(self.device)
                payoffs = [self.critics[i](obs, actions).cpu().numpy() for i in range(self.n_agents)]

                payoff_matrix[joint_action] = np.round(np.array(payoffs).squeeze(), 1)

        return payoff_matrix