import numpy as np
import random
from itertools import product
from copy import deepcopy

MAX_N_PLAYER = 64
FACT_ARR = np.array([2**i for i in range(MAX_N_PLAYER)])

def binary2action(actions):
    if type(actions) is list:
        actions = np.array(actions)
    arr = FACT_ARR[:len(actions)]
    binary_representation = np.sum(actions * arr)
    return binary_representation

def action2binary(num_action, n_players):
    result = [int(i) for i in bin(num_action)[2:]]
    result.reverse()
    while len(result) < n_players:
        result.append(0)
    return result

class NashEquilibriumJudger():
    def any_single_player_payoff(strategy, payoff_matrix:np.ndarray, action_space, player=None):
        """
        計算多玩家遊戲的期望收益
        
        Args:
            strategy (list of list): 每個玩家的動作概率分布, strategy[i]是第i個玩家的動作分布, 長度為各自的action數量
            payoff_matrix (np.ndarray): payoff矩陣, shape為 action_space + (n_player,)
            action_space (tuple of int): 每個玩家的動作數量
            player (int): 回傳指定player的expected payoff, 預設為None則回傳所有player的expected payoff
        
        Returns:
            np.ndarray: 長度為 n_player 的一維陣列，表示每個玩家的期望收益
        """
        # 初始化每位玩家的期望收益
        n_player = action_space[-1]
        expected_payoff = np.zeros(n_player)
        action_space = action_space[:len(action_space) - 1]
        
        # 遍歷所有可能的動作組合
        action_combinations = product(*[range(a) for a in action_space])
        
        for actions in action_combinations:
            # 計算該動作組合的聯合概率
            joint_probability = 1.0
            for player_idx, action in enumerate(actions):
                joint_probability *= strategy[player_idx][action]
            
            # 獲取該動作組合的收益並加權到每位玩家的期望值
            expected_payoff += joint_probability * payoff_matrix[actions]

        if player is None:
            return expected_payoff
        return expected_payoff[player]
    
    def run_indifference(strategy:list, payoff_matrix:np.ndarray, epsilon=1e-8):
        """
        使用Indifference的方法判斷給定的策略是否為Nash Equilibrium
        
        Args:
            strategy (list): 表示玩家的策略, 其中玩家人數等於len(strategy)
            game (GeneralSumGame): payoff matrix
            epsilon (float): Mixed Strategy的Expected Payoff容忍值

        Returns:
            boolean: 是否為Nash Equilibrium
        """
        n_player = len(strategy)
        action_space = payoff_matrix.shape
        action_space = action_space[:len(action_space) - 1]
        # 遍歷每個玩家，檢查是否為其最佳反應
        for target_player in range(n_player):
            # 固定其他玩家的策略，計算目標玩家的每個動作的期望收益
            other_players = [i for i in range(n_player) if i != target_player]
            target_action_count = action_space[target_player]
            
            # 計算目標玩家每個動作的期望收益
            action_payoffs = np.zeros(target_action_count)
            for target_action in range(target_action_count):
                # 模擬目標玩家選擇 target_action
                joint_probability = 0.0
                for actions in product(*[range(a) for a in action_space]):
                    if actions[target_player] != target_action:
                        continue
                    prob = 1.0
                    for player_idx, action in enumerate(actions):
                        if player_idx == target_player:
                            prob *= 1.0 if action == target_action else 0.0
                        else:
                            prob *= strategy[player_idx][action]
                    action_payoffs[target_action] += prob * payoff_matrix[actions][target_player]
            
            # 計算目標玩家的策略選擇的期望收益
            target_strategy_payoff = np.dot(strategy[target_player], action_payoffs)
            
            # 檢查是否每個動作的期望收益 <= 當前策略選擇的期望收益
            if not all(action_payoff <= target_strategy_payoff + epsilon for action_payoff in action_payoffs):
                # 當策略不是 Nash 均衡時，計算最佳回應
                better_strategy = np.zeros(target_action_count)
                best_action_idx = np.argmax(action_payoffs)
                better_payoff = action_payoffs[best_action_idx]
                better_strategy[best_action_idx] = 1.0  # 將最優動作設為 1（純策略最佳回應）
                return False, {'Player': 'Player '+str(target_player+1), 'better strategy': better_strategy, 
                               'New Payoff': better_payoff, 'Old Payoff': target_strategy_payoff}
        
        return True, {}
        # n_player = len(strategy)
        # action_space = payoff_matrix.shape
        # info = {}

        # #is_nash = True
        # other_player_strategy = []
        # for player in range(n_player):
        #     other_player_strategy.append(np.ones(action_space[player]))

        # for player in range(n_player):
        #     n_actions = action_space[player]

        #     # Pure Strategy
        #     if 1.0 in strategy[player]:
        #         info['Judge'] = 'Pure'
        #         action_idx = np.where(strategy[player]==1.0)[0][0]
        #         expected_payoffs = np.zeros(n_actions)
        #         for a in range(n_actions):
        #             strategy_copy = deepcopy(strategy)
        #             player_strategy = np.zeros(n_actions)
        #             player_strategy[a] = 1.0
        #             strategy_copy[player] = player_strategy
        #             expected_payoffs[a] = NashEquilibriumJudger.any_single_player_payoff(strategy_copy, payoff_matrix, action_space, player)
        #         best_action_idx = np.argmax(expected_payoffs)
        #         if best_action_idx != action_idx:
        #             info['Player'] = 'Player ' + str(player + 1)
        #             info['Payoff'] = expected_payoffs
        #             return False, info
        #     else:
        #     # Mixed Strategy
        #         info['Judge'] = 'Mixed'
        #         expected_payoffs = np.zeros(n_actions)
        #         prev_expected_payoff = None
                
        #         for a in range(n_actions):
        #             if not np.isclose(strategy[player][a], 0.0):
        #                 strategy_copy = deepcopy(other_player_strategy)
        #                 player_strategy = np.zeros(n_actions)
        #                 player_strategy[a] = strategy[player][a]
        #                 strategy_copy[player] = player_strategy
        #                 expected_payoffs[a] = NashEquilibriumJudger.any_single_player_payoff(strategy_copy, payoff_matrix, action_space, player)
        #                 if prev_expected_payoff is not None:
        #                     if not np.isclose(expected_payoffs[a], prev_expected_payoff):
        #                         info['Player'] = 'Player ' + str(player + 1)
        #                         info['Payoff'] = expected_payoffs
        #                         return False, info
        #                 else:
        #                     prev_expected_payoff = expected_payoffs[a]
        # info['Payoff'] = NashEquilibriumJudger.any_single_player_payoff(strategy, payoff_matrix, action_space)
        # return True, info

    def run(strategy, payoff_matrix, n_steps=int(10e5), epsilon=0.0, method='random', matrix_type='normal'):
        n_player = strategy.shape[0]
        original_payoffs = NashEquilibriumJudger.get_payoff(strategy, payoff_matrix, matrix_type)
        
        # 對於每一位玩家，逐步檢查是否可以通過改變策略增加收益
        is_nash = True
        info = {'New Payoff': -np.inf}
        for player in range(n_player):
            if method == 'random' and is_nash:
                for _ in range(n_steps):
                    # 生成新的隨機策略分布
                    a0 = random.random()
                    a1 = 1.0 - a0
                    new_strategy = strategy.copy()
                    new_strategy[player] = np.array([a0, a1])
                    
                    # 計算該玩家新的期望收益
                    new_payoff = NashEquilibriumJudger.calculate_single_player_payoff(player, new_strategy, payoff_matrix, matrix_type)
                    
                    # 檢查該玩家是否在新策略下得到了更高的收益
                    if new_payoff > original_payoffs[player] * ((1 + epsilon) if original_payoffs[player] > 0.0 else (1 - epsilon)):
                        is_nash = False
                        if info['New Payoff'] < new_payoff:
                            target_player = 'Player ' + str(player + 1)
                            info[target_player] = [a0, a1]
                            info['Player'] = target_player
                            info['New Payoff'] = new_payoff
                        #return False,  info # 若找到能提高收益的策略，則非 Nash Equilibrium
            
        # 若所有測試均未找到更高收益的策略，則認為是 Nash Equilibrium
        return is_nash, info
            
    def calculate_single_player_payoff(player, strategy, payoff_matrix, matrix_type='normal'):
        """只計算單個玩家的期望收益"""
        n_player = strategy.shape[0]
        expected_payoff = 0.0

        # 生成所有其他玩家的行為組合
        other_actions = list(product([0, 1], repeat=n_player - 1))

        # 遍歷所有其他玩家的行為組合
        for actions in other_actions:
            for player_action in [0, 1]:  # 該玩家的兩種行為
                # 構建完整的行為組合
                full_action = list(actions)
                full_action.insert(player, player_action)

                # 計算該組合的機率
                prob = strategy[player][player_action]
                for i, action in enumerate(full_action):
                    if i != player:
                        prob *= strategy[i][action]

                # 從收益矩陣中獲取該行為組合下該玩家的收益
                payoff = payoff_matrix[tuple(full_action)][player] \
                        if matrix_type=='normal' else\
                        payoff_matrix[action2binary(full_action)][player]

                # 根據機率加權收益
                expected_payoff += prob * payoff

        return expected_payoff

    def get_payoff(strategy, payoff_matrix, matrix_type='normal'):
        n_players = strategy.shape[0]
        expected_payoffs = np.zeros(n_players)  # 初始化每個玩家的期望收益

        # 生成所有可能的行為組合，這裡使用 (0, 0, ..., 0) 到 (1, 1, ..., 1)
        all_actions = list(product([0, 1], repeat=n_players))

        # 遍歷每個行為組合
        for actions in all_actions:
            # 計算該行為組合的機率
            prob = 1.0
            for i, action in enumerate(actions):
                prob *= strategy[i][action]

            # 從收益矩陣中獲取該行為組合的收益
            payoffs = payoff_matrix[actions]\
                    if matrix_type=='normal' else\
                    payoff_matrix[action2binary(actions)]
            
            # 根據該行為組合的機率加權收益
            expected_payoffs += prob * payoffs

        return expected_payoffs


if __name__ == '__main__':
    #print(action2binary([1 for i in range(100)]))
    #print(binary2actions(12, 5))
    # action_space = np.zeros((2, 4, 1, 3)).shape
    # action_space = action_space[:len(action_space) -1 ]
    # print(np.isclose(action_space[0], 2.0))
    prisoner_dilemma = np.array([
        [[3, 3], [0, 5]],
        [[5, 0], [1, 1]]
    ])
    import pygambit as gbt
    from game_generator import *
    run_pass_game = np.array([
        [[0, 0], [10, -10]],
        [[5, -5], [0, 0]]
    ])
    Penalty_Kick_game = np.array([
        [[0.58,-0.58], [0.95,-0.95]],
        [[0.93,-0.93], [0.70,-0.70]]
    ])
    key_note_game = np.array([
        [[1, 1], [0, 0]],
        [[0, 0], [2, 2]]
    ])
    game = GeneralSumGame(2, Penalty_Kick_game)
    result = gbt.nash.lcp_solve(game.gamebit_form, rational=True, stop_after=1).equilibria
    print(result)
    strategy = np.array([
        [0.0, 1.0],  # Player 1 的策略
        [0.0, 1.0]   # Player 2 的策略
        # prisoner_dilemma
    ])
    # strategy = np.array([
    #     [1.0/3.0, 2.0/3.0],  # Player 1 的策略
    #     [2.0/3.0, 1.0/3.0]   # Player 2 的策略
    #     # run pass game
    # ])
    strategy = np.array([
        [23.0/60.0, 37.0/60.0],  # Player 1 的策略
        [5.0/12.0, 7.0/12.0]   # Player 2 的策略
        # Penalty_Kick
    ])
    #expected_payoff = NashEquilibriumJudger.get_payoff(strategy, payoff_matrix)
    #print(NashEquilibriumJudger.run(strategy, payoff_matrix, n_steps=100000, epsilon=0.01))
    print(strategy)
    print(NashEquilibriumJudger.run_indifference(strategy, game.payoff_matrix))