import numpy as np
from itertools import product
import random
import csv
from team_feasibility import feasibility_run
from game_generator import TwoTeamZeroSumSymmetricGame, TwoTeamSymmetricGame
from judger import NashEquilibriumJudger
from torch.utils.tensorboard import SummaryWriter

class StochasticGame():
    def __init__(self, n_states:int, n_agents:int, n_actions:int, payoff_matrix:np.ndarray = None, state_transmition:np.ndarray = None, terminal_state:list = None, seed=None, noise=None):
        """
            Payoff Matrix Format: (n_states, n_actions,...,n_actions, n_agents)
            first axis represents the stochastic games has n_states
            last axis represents the payoff for each agent under certain joint action

            State Transmition Function T: (n_states, n_states)
            probability only depends on state, and last axis is the probability, which means sum(T[s]) is 1.0 for any s in states
        """
        self.noise = noise
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
        if self.noise is not None:
            joint_rewards = joint_rewards + self.noise.sample(size=self.n_agents)

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
        state_transmition = None if payoff_matrix is None else np.ones((1, 1))
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
    def __init__(self, n_states:int, n_agents:int, n_actions:int, payoff_matrix:np.ndarray = None, state_transmition:np.ndarray = None, terminal_state:list = None, seed=None, noise=None):
        """
            Payoff Matrix Format: (n_states, n_actions,...,n_actions, n_agents)
            first axis represents the stochastic games has n_states
            last axis represents the payoff for each agent under certain joint action

            State Transmition Function T: (n_states, n_states)
            probability only depends on state, and last axis is the probability, which means sum(T[s]) is 1.0 for any s in states
        """
        self.noise = noise
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

    def read_matrix(self, file_name) -> np.ndarray:
        payoff_matrix = np.zeros((2, 2, 2, 2, 4), dtype=int)

        with open(file_name, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        p4 = None
        p3 = None
        i = 0

        while i < len(lines):
            line = lines[i]

            # Player 4 header
            if line.startswith("Player 4:"):
                p4 = int(line.split(":")[1].strip())
                i += 1
                continue

            # Player 3 和 Player 2 header
            if line.startswith("Player 3:"):
                parts = line.split(",")
                p3 = int(parts[0][-1])
                p2_0 = int(parts[1][-1])
                p2_1 = int(parts[2][-1])
                p2_headers = [p2_0, p2_1]
                i += 1

                # Player 1 的兩行
                for p1 in range(2):
                    line_data = lines[i][12:].split("\",\"")
                    # if len(line_data) < 3:
                    #     i += 1
                    #     continue
                    payoff0, payoff1 = line_data
                    payoff0, payoff1 = payoff0[1:], payoff1[:-1]
                    for p2, payoff_str in zip(p2_headers, [payoff0, payoff1]):
                        payoff = list(map(int, payoff_str.split(",")))
                        payoff_matrix[p1, p2, p3, p4] = payoff
                    i += 1
                continue

            i += 1

        return payoff_matrix
    
    def from_csv(self, file_name) -> StochasticGame:
        payoff_matrix = np.zeros((1,2,2,2,2,4))
        payoff_matrix[0] = self.read_matrix(file_name)

        return TwoTeamZeroSumSymmetricStochasticEnv(n_states=self.n_states, n_agents=self.n_agents, n_actions=self.n_actions,
                                             payoff_matrix=payoff_matrix, state_transmition=np.ones((1, 1)), terminal_state=[0])

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
    
    def save(self, infos=[], config={}):
        '''
        save following infomation to .csv(4 players only)
        payoff matrix
        (feasibility)nash equilibrium and its expected payoff
        Q-values for all agents and alogrithms
        Parameters:
        - infos (list): a list of dictionaries, which contains alogrithm name and q-values with keys \'algo\' and \'q-values\'
        '''
        if self.n_agents == 4 and self.n_actions == 2:
            valid = True
            logdir = config['logdir']
            logger = SummaryWriter('./runs/'+logdir)
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
                game = self.feasibility_game()
                strategy, _, _ = feasibility_run(game)
                if np.any(strategy < 0.0):
                    valid = False
                strategy = strategy.reshape(self.n_agents, 2)
                expected_payoff = NashEquilibriumJudger.get_payoff(strategy, game.payoff_matrix)
                player_strategy_str = []
                for i in range(self.n_agents):
                    player_strategy_str.append('[' + ', '.join(str(round(x, 4)) for x in strategy[i]) + ']')
                writer.writerow(player_strategy_str)
                infos.insert(0, {'algo':'feasibility',
                                'expected_payoff':expected_payoff})
                
                # nash eqilibrium expected payoff asymptote
                for i in range(config['batch_size']-1, config['total_steps']):
                    logger.add_scalar('Train/Team1-Ep.Reward', expected_payoff[ 0], i+1)
                    logger.add_scalar('Train/Team2-Ep.Reward', expected_payoff[-1], i+1)
                    break

                # q-values
                for algo_info in infos:
                    writer.writerow([])
                    writer.writerow([algo_info['algo']])
                    writer.writerow([''] + ['Player '+str(i+1) for i in range(self.n_agents)])
                    q_values = []
                    max_qs = []
                    for i in range(self.n_agents):
                        expected_payoff = np.dot(strategy[i], algo_info['q-values'][i]) if algo_info['algo']!='feasibility' else algo_info['expected_payoff'][i]
                        q_values.append(str(expected_payoff.round(3)))
                        max_qs.append(str(np.max(algo_info['q-values'][i].round(3))) if algo_info['algo']!='feasibility' else '')
                    writer.writerow(['dot'] + q_values)
                    writer.writerow(['max'] + max_qs)
            return valid

    def feasibility_game(self):
        game = TwoTeamZeroSumSymmetricGame(n_players=self.n_agents)
        game.set_payoff_matrix(self.payoff_matrix[0])
        return game
                
class TwoTeamSymmetricStochasticEnv(TwoTeamZeroSumSymmetricStochasticEnv):
    def __init__(self, n_states:int, n_agents:int, n_actions:int, payoff_matrix:np.ndarray = None, state_transmition:np.ndarray = None, terminal_state:list = None, seed=None, noise=None):
        """
            Payoff Matrix Format: (n_states, n_actions,...,n_actions, n_agents)
            first axis represents the stochastic games has n_states
            last axis represents the payoff for each agent under certain joint action

            State Transmition Function T: (n_states, n_states)
            probability only depends on state, and last axis is the probability, which means sum(T[s]) is 1.0 for any s in states
        """
        self.noise = noise
        self.low  =  0
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
            self.team_1_payoff_indicator = []
            self.team_2_payoff_indicator = []
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
        
    def from_csv(self, file_name) -> StochasticGame:
        payoff_matrix = np.zeros((1,2,2,2,2,4))
        payoff_matrix[0] = self.read_matrix(file_name)

        return TwoTeamSymmetricStochasticEnv(n_states=self.n_states, n_agents=self.n_agents, n_actions=self.n_actions,
                                             payoff_matrix=payoff_matrix, state_transmition=np.ones((1, 1)), terminal_state=[0])

    def generate_random_game(self) -> np.ndarray:
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = np.zeros(shape=shape, dtype=np.int8)

        self.joint_action_indicator.append(np.random.randint(low=self.low, high=self.high+1, size=(self.n_players//2+1, self.n_players//2+1, 2), dtype=np.int8))
        
        self.team_1_payoff_indicator.append(np.arange(self.low, self.high+1, dtype=np.int8))
        self.team_2_payoff_indicator.append(np.arange(self.low, self.high+1, dtype=np.int8))

        random.shuffle(self.team_1_payoff_indicator[self.state])
        random.shuffle(self.team_2_payoff_indicator[self.state])

        all_actions = list(product([0, 1], repeat=self.n_players))
        for joint_action in all_actions:
            team_1_action_indicator = joint_action[:self.n_players//2].count(1)
            team_2_action_indicator = joint_action[self.n_players//2:].count(1)
            joint_action_idx = self.joint_action_indicator[self.state][team_1_action_indicator, team_2_action_indicator]
            team_1_payoffs = self.team_1_payoff_indicator[self.state][joint_action_idx[0]]
            team_2_payoffs = self.team_2_payoff_indicator[self.state][joint_action_idx[1]]
            payoffs = np.concatenate((np.full(shape=self.n_players//2, fill_value=team_1_payoffs, dtype=np.int8),\
                                      np.full(shape=self.n_players//2, fill_value=team_2_payoffs, dtype=np.int8)))
            payoff_matrix[joint_action] = payoffs
            
        return payoff_matrix
    
    def get_u(self, k:int, l:int) -> np.ndarray:
        joint_action_idx = self.joint_action_indicator[self.state][k, l]
        team1_payoffs = self.team_1_payoff_indicator[self.state][joint_action_idx[0]]
        team2_payoffs = self.team_2_payoff_indicator[self.state][joint_action_idx[1]]
        return np.array([team1_payoffs, team2_payoffs])
    
    def feasibility_game(self):
        game = TwoTeamSymmetricGame(n_players=self.n_agents)
        game.set_payoff_matrix(self.payoff_matrix[0])
        return game
    
class GMP(TwoTeamZeroSumSymmetricStochasticEnv):
    def __init__(self, w=0.5):
        self.n_players = 4
        payoff_matrix = np.zeros((1, 2, 2, 2, 2, 4), dtype=np.float16)
        all_actions = list(product([0, 1], repeat=self.n_players))
        for joint_action in all_actions:
            team_1_H = joint_action[:self.n_players//2].count(1)
            team_2_H = joint_action[self.n_players//2:].count(1)
            if team_1_H == 2:
                if team_2_H == 2:
                    payoff = np.array([1, 1, -1, -1], dtype=np.float16)
                elif team_2_H == 0:
                    payoff = np.array([-1, -1, 1, 1], dtype=np.float16)
                else:
                    payoff = np.array([w, w, -w, -w], dtype=np.float16)
            elif team_1_H == 0:
                if team_2_H == 2:
                    payoff = np.array([-1, -1, 1, 1], dtype=np.float16)
                elif team_2_H == 0:
                    payoff = np.array([1, 1, -1, -1], dtype=np.float16)
                else:
                    payoff = np.array([w, w, -w, -w], dtype=np.float16)
            else:
                if team_2_H == 1:
                    payoff = np.array([0, 0, 0, 0], dtype=np.float16)
                else:
                    payoff = np.array([-w, -w, w, w], dtype=np.float16)

            payoff_matrix[0][joint_action] = payoff

        super().__init__(n_states=1, n_agents=4, n_actions=2, payoff_matrix=payoff_matrix,\
                         state_transmition=np.ones((1,1)), terminal_state=[0])

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