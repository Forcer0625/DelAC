import numpy as np
from itertools import product
import random
import csv
from team_feasibility import feasibility_run
from game_generator import TwoTeamZeroSumSymmetricGame
from judger import NashEquilibriumJudger

class StochasticGame():
    def __init__(self, n_states:int, n_agents:int, n_actions:int, payoff_matrix:np.ndarray = None, state_transmition:np.ndarray = None, terminal_state:list = None, seed=None):
        """
            Payoff Matrix Format: (n_states, n_actions,...,n_actions, n_agents)
            first axis represents the stochastic games has n_states
            last axis represents the payoff for each agent under certain joint action

            State Transmition Function T: (n_states, n_states)
            probability only depends on state, and last axis is the probability, which means sum(T[s]) is 1.0 for any s in states
        """
        self.low = 0
        self.high = 10
        self.max_step = 100
        np.random.seed(seed)
        self.n_states = n_states
        self.n_players = self.n_agents = n_agents
        self.n_actions = n_actions
        if payoff_matrix is None or state_transmition is None:
            self.state_transmition = np.zeros((n_states, n_states))
            shape = [n_actions for _ in range(self.n_agents)]
            shape.append(n_agents)
            shape.insert(0, n_states)
            self.payoff_matrix = np.zeros(shape, dtype=np.int8)
            for s in range(n_states):
                self.state_transmition[s] = np.random.dirichlet(np.ones(n_states), size=1) # generate random state transmition probability
                self.payoff_matrix[s] = self.generate_random_game()
            self.terminal_state = list(np.random.choice([i for i in range(n_states)], size=1))
        else:
            assert payoff_matrix.shape[ 0] == n_states == state_transmition.shape[0]
            assert payoff_matrix.shape[ 1] == n_actions
            assert payoff_matrix.shape[-1] == n_agents
            self.payoff_matrix = payoff_matrix
            self.state_transmition = state_transmition
            self.terminal_state = terminal_state

    def step(self, joint_actions:np.ndarray|list):
        info = {
            'actions':joint_actions,
            'prev_state':self.state,
            'state_transmition_probs':self.state_transmition[self.state],
        }
        self.n_steps += 1

        # 1. get payoff
        joint_rewards = self.payoff_matrix[self.state][tuple(joint_actions)]

        # 2. state transmition
        self.state = np.random.choice([i for i in range(self.n_states)], size=1, p=self.state_transmition[self.state])[0]

        # 3. others
        terminiation = self.state in self.terminal_state
        truncation = self.max_step <= self.n_steps
        info['next_state'] = self.state
        
        return self.make_state(), joint_rewards, terminiation, truncation, info

    def reset(self, ):
        self.state = 0 # initial state is 0(fixed)
        self.n_steps = 0
        return self.make_state(), {}

    def generate_random_game(self) -> np.ndarray:
        rng = np.random.default_rng()
        shape = [self.n_actions for _ in range(self.n_agents)]
        shape.append(self.n_agents)
        payoff_matrix = rng.integers(low=self.low, high=self.high, endpoint=True, size=shape, dtype=np.int8)
        return payoff_matrix
    
    def make_state(self):
        return np.full((self.n_agents, 1), fill_value=self.state)

class NormalFormGame(StochasticGame):
    def __init__(self, n_agents:int, n_actions:int, payoff_matrix:np.ndarray = None, seed=None):
        if payoff_matrix is not None:
            assert payoff_matrix.shape[ 0] == n_actions
            assert payoff_matrix.shape[-1] == n_agents
            payoff_matrix = np.expand_dims(payoff_matrix, axis=0)
        state_transmition = None if payoff_matrix is None else np.ones(1, 1)
        terminal_state = None if payoff_matrix is None else [0]
        super().__init__(n_states=1, n_agents=n_agents, n_actions=n_actions,\
                         payoff_matrix=payoff_matrix, state_transmition=state_transmition,\
                         terminal_state=terminal_state, seed=seed)

    def step(self, actions:np.ndarray):
        """return obs, reward, termination, truncation, infos"""
        payoff = self.payoff_matrix[0][tuple(actions)] # all agents payoff
        return np.zeros((self.n_agents, 1)), payoff, True, False, {'actions':actions}

    def reset(self):
        """return obs, infos"""
        return np.zeros((self.n_agents, 1)), {}

class TwoTeamZeroSumSymmetricEnv(NormalFormGame):
    def generate_random_game(self) -> np.ndarray:
        self.low = -10
        self.high = 10
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = np.zeros(shape=shape, dtype=np.int8)

        self.joint_action_indicator = np.random.randint(low=self.low, high=self.high+1, size=(self.n_players//2+1, self.n_players//2+1), dtype=np.int8)
        
        self.team_payoff_indicator = np.arange(self.low, self.high+1, dtype=np.int8)

        random.shuffle(self.team_payoff_indicator)

        all_actions = list(product([0, 1], repeat=self.n_players))
        for joint_action in all_actions:
            team_1_action_indicator = joint_action[:self.n_players//2].count(1)
            team_2_action_indicator = joint_action[self.n_players//2:].count(1)
            joint_action_idx = self.joint_action_indicator[team_1_action_indicator, team_2_action_indicator]
            team_payoffs = self.team_payoff_indicator[joint_action_idx]
            payoffs = np.concatenate((np.full(shape=self.n_players//2, fill_value= team_payoffs, dtype=np.int8),\
                                      np.full(shape=self.n_players//2, fill_value=-team_payoffs, dtype=np.int8)))
            payoff_matrix[joint_action] = payoffs
            
        return payoff_matrix
    
    def get_u(self, k:int, l:int) -> np.ndarray:
        joint_action_idx = self.joint_action_indicator[k, l]
        team1_payoffs = self.team_payoff_indicator[joint_action_idx]
        team2_payoffs = -team1_payoffs
        return np.array([team1_payoffs, team2_payoffs])

class TwoTeamZeroSumSymmetricStochasticEnv(StochasticGame):
    def __init__(self, n_states:int, n_agents:int, n_actions:int, payoff_matrix:np.ndarray = None, state_transmition:np.ndarray = None, terminal_state:list = None, seed=None):
        """
            Payoff Matrix Format: (n_states, n_actions,...,n_actions, n_agents)
            first axis represents the stochastic games has n_states
            last axis represents the payoff for each agent under certain joint action

            State Transmition Function T: (n_states, n_states)
            probability only depends on state, and last axis is the probability, which means sum(T[s]) is 1.0 for any s in states
        """
        self.low = -10
        self.high = 10
        self.max_step = 100
        np.random.seed(seed)
        self.n_states = n_states
        self.n_players = self.n_agents = n_agents
        self.n_actions = n_actions
        if payoff_matrix is None or state_transmition is None:
            self.state_transmition = np.zeros((n_states, n_states))
            shape = [n_actions for _ in range(self.n_agents)]
            shape.append(n_agents)
            shape.insert(0, n_states)
            self.payoff_matrix = np.zeros(shape, dtype=np.int8)
            self.joint_action_indicator = []
            self.team_payoff_indicator = []
            for s in range(n_states):
                self.state = s
                self.state_transmition[s] = np.random.dirichlet(np.ones(n_states), size=1) # generate random state transmition probability
                self.payoff_matrix[s] = self.generate_random_game()
            self.terminal_state = list(np.random.choice([i for i in range(n_states)], size=1))
            self.state = 0
        else:
            assert payoff_matrix.shape[ 0] == n_states == state_transmition.shape[0]
            assert payoff_matrix.shape[ 1] == n_actions
            assert payoff_matrix.shape[-1] == n_agents
            self.payoff_matrix = payoff_matrix
            self.state_transmition = state_transmition
            self.terminal_state = terminal_state

    def generate_random_game(self) -> np.ndarray:
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = np.zeros(shape=shape, dtype=np.int8)

        self.joint_action_indicator.append(np.random.randint(low=self.low, high=self.high+1, size=(self.n_players//2+1, self.n_players//2+1), dtype=np.int8))
        
        self.team_payoff_indicator.append(np.arange(self.low, self.high+1, dtype=np.int8))

        random.shuffle(self.team_payoff_indicator[self.state])

        all_actions = list(product([0, 1], repeat=self.n_players))
        for joint_action in all_actions:
            team_1_action_indicator = joint_action[:self.n_players//2].count(1)
            team_2_action_indicator = joint_action[self.n_players//2:].count(1)
            joint_action_idx = self.joint_action_indicator[self.state][team_1_action_indicator, team_2_action_indicator]
            team_payoffs = self.team_payoff_indicator[self.state][joint_action_idx]
            payoffs = np.concatenate((np.full(shape=self.n_players//2, fill_value= team_payoffs, dtype=np.int8),\
                                      np.full(shape=self.n_players//2, fill_value=-team_payoffs, dtype=np.int8)))
            payoff_matrix[joint_action] = payoffs
            
        return payoff_matrix
    
    def get_u(self, k:int, l:int) -> np.ndarray:
        joint_action_idx = self.joint_action_indicator[self.state][k, l]
        team1_payoffs = self.team_payoff_indicator[self.state][joint_action_idx]
        team2_payoffs = -team1_payoffs
        return np.array([team1_payoffs, team2_payoffs])
    
    def save(self, infos=[], logdir='example'):
        '''
        save following infomation to .csv(4 players only)
        payoff matrix
        (feasibility)nash equilibrium and its expected payoff
        Q-values for all agents and alogrithms
        Parameters:
        - infos (list): a list of dictionaries, which contains alogrithm name and q-values with keys \'algo\' and \'q-values\'
        '''
        if self.n_agents == 4 and self.n_actions == 2:
            with open('./runs/'+logdir+'/data.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for player4_action in [0, 1]:
                    writer.writerow(['Player 4: '+str(player4_action)])
                    for player3_action in [0, 1]:
                        writer.writerow(['Player 3: '+str(player3_action), 'Player 2: 0', 'Player 2: 1'])
                        for player1_action in [0, 1]:
                            write_data = ['Player 1: '+str(player1_action)]
                            for player2_acton in [0, 1]:
                                joint_action = (player1_action, player2_acton, player3_action, player4_action)
                                pay_off_str = ', '.join(str(x) for x in self.payoff_matrix[0][joint_action])
                                write_data.append(pay_off_str)
                            writer.writerow(write_data)
                        writer.writerow([])
                    writer.writerow([])

                # feasibility
                game = TwoTeamZeroSumSymmetricGame(n_players=self.n_agents)
                game.set_payoff_matrix(self.payoff_matrix[0])
                strategy, _, _ = feasibility_run(game)
                strategy = strategy.reshape(self.n_agents, 2)
                expected_payoff = NashEquilibriumJudger.get_payoff(strategy, game.payoff_matrix)
                player_strategy_str = []
                for i in range(self.n_agents):
                    player_strategy_str.append('[' + ', '.join(str(round(x, 4)) for x in strategy[i]) + ']')
                writer.writerow(player_strategy_str)
                infos.insert(0, {'algo':'feasibility',
                                'q-values':expected_payoff})

                # q-values
                for algo_info in infos:
                    writer.writerow([])
                    writer.writerow([algo_info['algo']])
                    writer.writerow(['Player '+str(i+1) for i in range(self.n_agents)])
                    q_values = []
                    for q_value in list(algo_info['q-values'].round(3)):
                        q_values.append(str(q_value))
                    writer.writerow(q_values)

if __name__ == '__main__':
    game = TwoTeamZeroSumSymmetricEnv(n_agents=4, n_actions=2)
    a, b = game.reset()
    joint_actions = [0, 1, 1, 0]
    a, b, c, d, e = game.step(joint_actions)
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)