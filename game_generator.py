import numpy as np
import pygambit as gbt
from judger import action2binary, binary2action
from copy import deepcopy
from itertools import product
import random

def compute_two_player_game_payoff_corrected(n_players, payoff_matrix, player_idx):
    """
    將 n 玩家博弈的收益矩陣高效轉換為 i-th 兩人博弈的收益矩陣，確保符合等式(3)中小到大的順序。

    Parameters:
    - n_players (int): 玩家數量。
    - payoff_matrix (ndarray): 原始 n 玩家博弈的收益矩陣，形狀為 (2, 2, ..., 2, n_players)。
    - player_idx (int): 目標玩家的索引 (0 ~ n_players-1)。

    Returns:
    - two_player_game_payoff_matrix (ndarray): 
      形狀為 (2, 2^(n_players-1), 2)，
      第一維為玩家 1 的動作，第二維為玩家 2 的聚合動作，
      第三維分別表示兩位玩家的收益。
    """
    # 確定動作空間大小
    num_joint_actions = 2 ** n_players  # 總的 joint action 數
    num_player2_actions = 2 ** (n_players - 1)  # Player 2 聚合動作數
    
    # 展平收益矩陣以方便操作
    flat_payoff_matrix = payoff_matrix.reshape(num_joint_actions, n_players)
    
    # 玩家 1 的收益
    player1_payoff = flat_payoff_matrix[:, player_idx]  # 玩家 1 的收益
    # 玩家 2 的收益
    player2_payoff = np.sum(flat_payoff_matrix, axis=1) - player1_payoff  # 其他玩家的收益總和
    
    # 解碼所有 joint action 為二進制
    joint_action_indices = np.arange(num_joint_actions)  # 所有 joint action 的索引
    joint_action_bits = np.unpackbits(
        joint_action_indices[:, None].astype(np.uint8),
        axis=1,
        count=n_players,
        bitorder='little'
    )  # 二進制表示，形狀為 (num_joint_actions, n_players)
    
    # 提取 Player 1 和 Player 2 的動作
    player1_actions = joint_action_bits[:, player_idx]  # Player 1 的動作
    player2_actions_bits = np.delete(joint_action_bits, player_idx, axis=1)  # 其他玩家的動作

    # 將其他玩家的動作按位計算 Player 2 聚合動作的索引
    player2_actions_indices = np.dot(player2_actions_bits, 1 << np.arange(n_players - 1))

    # 初始化結果矩陣
    two_player_game_payoff_matrix = np.zeros((2, num_player2_actions, 2))
    
    # 填入收益矩陣
    for player1_action in [0, 1]:
        # 選取 Player 1 的對應行為
        mask = player1_actions == player1_action
        two_player_game_payoff_matrix[player1_action, :, 0] = np.bincount(
            player2_actions_indices[mask], weights=player1_payoff[mask], minlength=num_player2_actions
        )
        two_player_game_payoff_matrix[player1_action, :, 1] = np.bincount(
            player2_actions_indices[mask], weights=player2_payoff[mask], minlength=num_player2_actions
        )
    
    return two_player_game_payoff_matrix

def compute_two_player_game_payoff(n_players, payoff_matrix, player_idx):
    """
    將 n 玩家博弈的收益矩陣高效轉換為 i-th 兩人博弈的收益矩陣。

    Parameters:
    - n_players (int): 玩家數量。
    - payoff_matrix (ndarray): 原始 n 玩家博弈的收益矩陣，形狀為 (2, 2, ..., 2, n_players)。
    - player_idx (int): 目標玩家的索引 (0 ~ n_players-1)。

    Returns:
    - two_player_game_payoff_matrix (ndarray): 
      形狀為 (2, 2^(n_players-1), 2)，
      第一維為玩家 1 的動作，第二維為玩家 2 的聚合動作，
      第三維分別表示兩位玩家的收益。
    """
    # 確定動作空間
    total_actions = 2 ** n_players  # 總的 joint action 數
    num_player2_actions = 2 ** (n_players - 1)  # 玩家 2 聚合動作數
    
    # 將 payoff_matrix 展平，將每個 joint action 與其 payoff 匹配
    flat_payoff_matrix = payoff_matrix.reshape(total_actions, n_players)
    
    # 分解 joint action 的收益
    player1_payoff = flat_payoff_matrix[:, player_idx]  # 玩家 1 的收益
    player2_payoff = np.sum(flat_payoff_matrix, axis=1) - player1_payoff  # 玩家 2 的收益（其他玩家總和）
    
    # 將 joint action 的索引拆分為玩家 1 和玩家 2 的動作
    joint_action_indices = np.arange(total_actions)  # 所有 joint action 的索引
    joint_action_bits = np.unpackbits(
        joint_action_indices[:, None].astype(np.uint8), 
        axis=1, 
        count=n_players, 
        bitorder='little'
    )  # 每個 joint action 的二進制表示
    player1_actions = joint_action_bits[:, player_idx]  # 玩家 1 的動作
    other_players_actions = np.delete(joint_action_bits, player_idx, axis=1)  # 其他玩家的動作
    player2_aggregated_actions = np.packbits(other_players_actions, axis=1, bitorder='little').flatten()  # 聚合後的玩家 2 動作
    
    # 構建玩家 1 和玩家 2 的收益矩陣
    two_player_game_payoff_matrix = np.zeros((2, num_player2_actions, 2))
    for action in [0, 1]:  # 玩家 1 的動作
        mask = player1_actions == action
        two_player_game_payoff_matrix[action, :, 0] = np.bincount(
            player2_aggregated_actions[mask], weights=player1_payoff[mask], minlength=num_player2_actions
        )
        two_player_game_payoff_matrix[action, :, 1] = np.bincount(
            player2_aggregated_actions[mask], weights=player2_payoff[mask], minlength=num_player2_actions
        )
    
    return two_player_game_payoff_matrix

def compute_two_player_game_payoff_(n_players, payoff_matrix, player_idx):
    """
    將 n 玩家博弈的收益矩陣轉換為 i-th 兩人博弈的收益矩陣。
    
    Parameters:
    - n_players (int): 玩家數量。
    - payoff_matrix (ndarray): 原始 n 玩家博弈的收益矩陣，形狀為 (2, 2, ..., 2, n_players)。
    - player_idx (int): 目標玩家的索引 (0 ~ n_players-1)。
    
    Returns:
    - two_player_game_payoff_matrix (ndarray): 
      形狀為 (2, 2^(n_players-1), 2)，第一維為玩家 1 的動作，第二維為玩家 2 的聚合動作，
      第三維分別表示兩位玩家的收益。
    """
    # 玩家 2 的動作空間大小為 2^(n_players-1)
    player2_actions = 2 ** (n_players - 1)
    # 初始化兩人博弈的收益矩陣
    two_player_game_payoff_matrix = np.zeros((2, player2_actions, 2), dtype=np.int8)
    
    # 遍歷玩家 1 的動作
    for a1 in [0, 1]:
        # 遍歷玩家 2 的動作 (聚合動作)
        for a2 in range(player2_actions):
            # 解析玩家 2 聚合動作為二進制
            binary_action = format(a2, f'0{n_players-1}b')
            joint_action = [int(b) for b in binary_action]
            joint_action.reverse()

            # 插入玩家 1 的動作到正確位置
            joint_action.insert(player_idx, a1)
            
            # 計算玩家 1 的收益
            v1 = payoff_matrix[tuple(joint_action)][player_idx]
            # 計算玩家 2 的收益 (其他玩家的收益總和)
            v2 = np.sum(payoff_matrix[tuple(joint_action)], dtype=np.int8) - v1
            
            # 存儲收益
            two_player_game_payoff_matrix[a1, a2, 0] = v1
            two_player_game_payoff_matrix[a1, a2, 1] = v2
    
    return two_player_game_payoff_matrix

def check(n_player:int, original_game:np.ndarray, two_player_game:np.ndarray, player_idx:int):
    """驗證 two_player_game 是否正確"""
    for player1_action in [0, 1]:
        for player2_action in range(2 ** (n_player - 1)):
            player2_action_binary = action2binary(player2_action, n_player - 1)
            joint_action = player2_action_binary.copy()
            joint_action.insert(player_idx, player1_action)
            joint_action = tuple(joint_action)

            # 檢查 Player 1 的收益
            if original_game[joint_action][player_idx] != two_player_game[player1_action, player2_action][0]:
                return False
            # 檢查 Player 2 的收益
            if original_game[joint_action].sum() - original_game[joint_action][player_idx] != \
                    two_player_game[player1_action, player2_action][1]:
                return False
    return True
            
class GeneralSumGame():
    def __init__(self, n_players, payoff_matrix=None, seed=None):
        if type(n_players) == str:
            pass # Read from file

        self.n_players = n_players
        if payoff_matrix is not None:
            assert n_players == payoff_matrix.shape[-1]
            self.payoff_matrix = payoff_matrix
        else:
            self.payoff_matrix = self.generator_game(seed)

        self.payoff_dict = {}
        for i in range(n_players):
            self.payoff_dict['Player '+str(i+1)] = self.payoff_matrix[..., i]

        self.gamebit_form = gbt.Game.from_dict(self.payoff_dict)
    
    def __len__(self):
        return self.n_players

    def __str__(self):
        return self.payoff_dict
    
    def constrained_two_player_approximation(self, player_idx, strategy):
        # 玩家 2 的動作空間大小為 2^(n_players - 1 - i)
        player2_actions = 2 ** (self.n_players - 1 - player_idx)
        # 初始化兩人博弈的收益矩陣
        two_player_game_payoff_matrix = np.zeros((2, player2_actions, 2), dtype=np.int8)

        # 遍歷玩家 1 的動作
        for a1 in [0, 1]:
            # 遍歷玩家 2 的動作 (聚合動作)
            for a2 in range(player2_actions):
                # 解析玩家 2 聚合動作為二進制
                binary_action = format(a2, f'0{self.n_players-1}b')
                joint_action = [int(b) for b in binary_action]
                joint_action.reverse()

                # 插入玩家 1 的動作到正確位置
                joint_action.insert(player_idx, a1)
                
                # 計算玩家 1 的收益
                v1 = self.payoff_matrix[tuple(joint_action)][player_idx]
                # 計算玩家 2 的收益 (剩餘其他玩家的收益總和)
                v2 = np.sum(self.payoff_matrix[tuple(joint_action)], dtype=np.int8) - \
                        self.payoff_matrix[tuple(joint_action)][:player_idx]
                
                # 儲存收益
                two_player_game_payoff_matrix[a1, a2, 0] = v1
                two_player_game_payoff_matrix[a1, a2, 1] = v2
        
        return GeneralSumGame.from_arrays(two_player_game_payoff_matrix)

    def two_virtual_player_approximation(self) -> None:
        """
        將 4 玩家博弈的收益矩陣轉換為 兩兩組人的收益矩陣。
        
        Parameters:
        - n_players (int): 玩家數量。
        - payoff_matrix (ndarray): 原始 n 玩家博弈的收益矩陣，形狀為 (2, 2, ..., 2, n_players)。
        - player_idx (int): 目標玩家的索引 (0 ~ n_players-1)。
        
        Returns:
        - two_player_game_payoff_matrix (ndarray): 
        形狀為 (2, 2^(n_players-1), 2)，第一維為玩家 1 的動作，第二維為玩家 2 的聚合動作，
        第三維分別表示兩位玩家的收益。
        """
        if self.n_players != 4:
            raise ValueError
        # 玩家 1、2 的動作空間大小為 4
        player1_actions = 4
        player2_actions = 4
        # 初始化兩人博弈的收益矩陣
        two_player_game_payoff_matrix = np.zeros((player1_actions, player2_actions, 2), dtype=np.int8)
        
        # 遍歷玩家 1 的動作
        for a1 in range(player1_actions):
            # 遍歷玩家 2 的動作 (聚合動作)
            for a2 in range(player2_actions):
                # 解析玩家 2 聚合動作為二進制
                binary_action1 = format(a1, f'0{self.n_players-2}b')
                joint_action1 = [int(b) for b in binary_action1]
                joint_action1.reverse()
                binary_action2 = format(a2, f'0{self.n_players-2}b')
                joint_action2 = [int(b) for b in binary_action2]
                joint_action2.reverse()

                joint_action = joint_action1 + joint_action2
                
                # 計算玩家 1 的收益
                v1 = np.sum(self.payoff_matrix[tuple(joint_action)][:2])
                # 計算玩家 2 的收益 (其他玩家的收益總和)
                v2 = np.sum(self.payoff_matrix[tuple(joint_action)][2:])
                
                # 存儲收益
                two_player_game_payoff_matrix[a1, a2, 0] = v1
                two_player_game_payoff_matrix[a1, a2, 1] = v2
        
        return GeneralSumGame.from_arrays(two_player_game_payoff_matrix)
    
    def two_player_approximation(self, player_idx) -> None:
        """
        將 n 玩家博弈的收益矩陣轉換為 i-th 兩人博弈的收益矩陣。
        
        Parameters:
        - n_players (int): 玩家數量。
        - payoff_matrix (ndarray): 原始 n 玩家博弈的收益矩陣，形狀為 (2, 2, ..., 2, n_players)。
        - player_idx (int): 目標玩家的索引 (0 ~ n_players-1)。
        
        Returns:
        - two_player_game_payoff_matrix (ndarray): 
        形狀為 (2, 2^(n_players-1), 2)，第一維為玩家 1 的動作，第二維為玩家 2 的聚合動作，
        第三維分別表示兩位玩家的收益。
        """
        # 玩家 2 的動作空間大小為 2^(n_players-1)
        player2_actions = 2 ** (self.n_players - 1)
        # 初始化兩人博弈的收益矩陣
        two_player_game_payoff_matrix = np.zeros((2, player2_actions, 2), dtype=np.int8)
        
        # 遍歷玩家 1 的動作
        for a1 in [0, 1]:
            # 遍歷玩家 2 的動作 (聚合動作)
            for a2 in range(player2_actions):
                # 解析玩家 2 聚合動作為二進制
                binary_action = format(a2, f'0{self.n_players-1}b')
                joint_action = [int(b) for b in binary_action]
                joint_action.reverse()

                # 插入玩家 1 的動作到正確位置
                joint_action.insert(player_idx, a1)
                
                # 計算玩家 1 的收益
                v1 = self.payoff_matrix[tuple(joint_action)][player_idx]
                # 計算玩家 2 的收益 (其他玩家的收益總和)
                v2 = np.sum(self.payoff_matrix[tuple(joint_action)], dtype=np.int8) - v1
                
                # 存儲收益
                two_player_game_payoff_matrix[a1, a2, 0] = v1
                two_player_game_payoff_matrix[a1, a2, 1] = v2
        
        return GeneralSumGame.from_arrays(two_player_game_payoff_matrix)
        debug = check(self.n_players, self.payoff_matrix, np.concatenate((player1_payoff_matrix, player2_payoff_matrix), axis=-1), player_idx)

        two_player_game_dict = {
            'Player 1': player1_payoff_matrix,
            'Player 2': player2_payoff_matrix
        }
        return gbt.Game.from_dict(two_player_game_dict)
    
    def from_arrays(payoff_matrix:np.ndarray):
        return GeneralSumGame(payoff_matrix.shape[-1], payoff_matrix)
    
    def from_dict(payoff_dict:dict):
        pass

    def read(filename:str):
        pass
        
    def generator_game(self, seed=None, low=0, high=10):
        rng = np.random.default_rng(seed)
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = rng.integers(low=low, high=high, endpoint=True, size=shape, dtype=np.int8)
        # if self.n_players == 3:
        #     payoff_matrix[..., 2] = payoff_matrix[..., 1] # v2 = v3
        # if self.n_players == 4:
        #     payoff_matrix[..., 1] = payoff_matrix[..., 0] # v1 = v2
        #     payoff_matrix[..., 3] = payoff_matrix[..., 2] # v3 = v4
        return payoff_matrix
    
    def get_u(self, k:int, l:int) -> np.ndarray:
        raise NotImplementedError
    
    def set_payoff_matrix(self, payoff_matrix, indicators):
        raise NotImplementedError
class TeamGame(GeneralSumGame):
    def generator_game(self, seed=None, low=0, high=10):
        rng = np.random.default_rng(seed)
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = rng.integers(low=low, high=high, endpoint=True, size=shape, dtype=np.int8)
        if self.n_players == 3:
            payoff_matrix[..., 2] = payoff_matrix[..., 1] # v2 = v3
        if self.n_players == 4:
            payoff_matrix[..., 3] = payoff_matrix[..., 2] = payoff_matrix[..., 1] = payoff_matrix[..., 0] # v1 = v2
            #payoff_matrix[..., 3] = payoff_matrix[..., 2] # v3 = v4
        return payoff_matrix

class SymmerticGame(TeamGame):
    def generator_game(self, seed=None, low=0, high=10):
        # rng = np.random.default_rng(seed)
        np.random.seed(seed)
        shape = [2 for _ in range(self.n_players)]
        all_actions = list(product([0, 1], repeat=self.n_players))
        shape.append(self.n_players)
        payoff_matrix = np.zeros(shape=shape, dtype=np.int8)
        payoff_indicator = np.random.randint(low=low, high=high+1, size=self.n_players+1, dtype=np.int8)
        # payoff_indicator[0] = payoff_indicator[-1] = 0
        for joint_action in all_actions:
            payoff_matrix[joint_action] = np.full(shape=self.n_players, \
                                                fill_value=payoff_indicator[joint_action.count(1)], dtype=np.int8)            
        return payoff_matrix
    
class TwoTeamSymmetricGame(GeneralSumGame):
    def generator_game(self, seed=None, low=0, high=10):
        # rng = np.random.default_rng(seed)
        np.random.seed(seed)
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = np.zeros(shape=shape, dtype=np.int8)

        self.joint_action_indicator = np.random.randint(low=low, high=high+1, size=(self.n_players//2+1, self.n_players//2+1, 2), dtype=np.int8)
        
        self.team_1_payoff_indicator = np.arange(low, high+1, dtype=np.int8)
        self.team_2_payoff_indicator = np.arange(low, high+1, dtype=np.int8)

        random.shuffle(self.team_1_payoff_indicator)
        random.shuffle(self.team_2_payoff_indicator)

        all_actions = list(product([0, 1], repeat=self.n_players))
        for joint_action in all_actions:
            team_1_action_indicator = joint_action[:self.n_players//2].count(1)
            team_2_action_indicator = joint_action[self.n_players//2:].count(1)
            joint_action_idx = self.joint_action_indicator[team_1_action_indicator, team_2_action_indicator]
            team_1_payoffs = self.team_1_payoff_indicator[joint_action_idx[0]]
            team_2_payoffs = self.team_2_payoff_indicator[joint_action_idx[1]]
            payoffs = np.concatenate((np.full(shape=self.n_players//2, fill_value=team_1_payoffs, dtype=np.int8),\
                                      np.full(shape=self.n_players//2, fill_value=team_2_payoffs, dtype=np.int8)))
            payoff_matrix[joint_action] = payoffs
            
        return payoff_matrix
    
    def get_u(self, k:int, l:int) -> np.ndarray:
        joint_action_idx = self.joint_action_indicator[k, l]
        team1_payoffs = self.team_1_payoff_indicator[joint_action_idx[0]]
        team2_payoffs = self.team_2_payoff_indicator[joint_action_idx[1]]
        return np.array([team1_payoffs, team2_payoffs])
    
class TwoTeamZeroSumSymmetricGame(TwoTeamSymmetricGame):
    def generator_game(self, seed=None, low=-10, high=10):
        # rng = np.random.default_rng(seed)
        np.random.seed(seed)
        shape = [2 for _ in range(self.n_players)]
        shape.append(self.n_players)
        payoff_matrix = np.zeros(shape=shape, dtype=np.int8)

        self.joint_action_indicator = np.random.randint(low=low, high=high+1, size=(self.n_players//2+1, self.n_players//2+1), dtype=np.int8)
        
        self.team_payoff_indicator = np.arange(low, high+1, dtype=np.int8)

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
    
    def set_payoff_matrix(self, payoff_matrix:np.ndarray):
        self.payoff_matrix = payoff_matrix
        self.joint_action_indicator = np.zeros((self.n_players//2+1, self.n_players//2+1), dtype=np.int32)
        min_payoff = np.min(payoff_matrix)
        max_payoff = np.max(payoff_matrix)
        self.team_payoff_indicator = np.arange(min_payoff, max_payoff+1, dtype=np.int32)

        all_actions = list(product([0, 1], repeat=self.n_players))
        for joint_action in all_actions:
            payoff = payoff_matrix[joint_action][0]
            team_1_action_indicator = joint_action[:self.n_players//2].count(1)
            team_2_action_indicator = joint_action[self.n_players//2:].count(1)
            self.joint_action_indicator[team_1_action_indicator, team_2_action_indicator] = payoff - min_payoff

if __name__ == "__main__":
    n_players = 4
    game = TwoTeamSymmetricGame(n_players)

    #for i in  range(n_players):
    two_player_game = game.two_virtual_player_approximation()
    exit()
    #print(check(n_players, game.payoff_matrix, two_player_game.payoff_matrix, player_idx=i))