import cvxpy as cp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from game_generator import GeneralSumGame, TeamGame
#from judger import NashEquilibriumJudger
from itertools import product

def compute_marginal_pmf(joint_dist):
    """
    Compute the marginal PMF for each random variable from a joint probability distribution.
    
    Parameters:
        joint_dist (np.ndarray): A 1D array of length 2^n representing the joint distribution.
                                 Each value in the array is >= 0 and np.sum(joint_dist) == 1.0.
    
    Returns:
        np.ndarray: A (n, 2) array where each row represents the marginal probabilities of a random variable.
                    The first column represents the probability of the variable being 0,
                    and the second column represents the probability of the variable being 1.
    """
    # Determine the number of random variables (n)
    n = int(np.log2(len(joint_dist)))
    
    # Ensure the input size is a power of 2 and the probabilities sum to 1
    if len(joint_dist) != 2**n or not np.isclose(np.sum(joint_dist), 1.0):
        raise ValueError("Input joint distribution must have length 2^n and sum to 1.")
    
    # Initialize an array to hold the marginal probabilities
    marginal_pmf = np.zeros((n, 2))
    
    # Iterate over each random variable
    for i in range(n):
        # Create masks for the current variable being 0 or 1
        mask_0 = [(idx >> i) & 1 == 0 for idx in range(len(joint_dist))]
        mask_1 = [(idx >> i) & 1 == 1 for idx in range(len(joint_dist))]
        
        # Compute the marginal probabilities for the current variable
        marginal_pmf[i, 0] = np.sum(joint_dist[mask_0])  # Probability of variable i being 0
        marginal_pmf[i, 1] = np.sum(joint_dist[mask_1])  # Probability of variable i being 1
    
    return list(marginal_pmf)

def get_all_strategy(x):
    #assert len(x) == 4
    x0, y0 = x[0], x[1]
    # Team 1
    x1 = 1.0 - x0
    # Team 2
    y1 = 1.0 - y0
    return np.array([x0, x1, x0, x1, y0, y1, y0, y1])

def get_max_u(game:GeneralSumGame, strategy):
    pass

def get_r(game:GeneralSumGame, strategy):
    pass

def calculate_expected_payoff(player_idx, game:GeneralSumGame, strategy):
    """
    計算遊戲理論中某個玩家的期望payoff
    
    Args:
        player_idx (int): 表示第幾個玩家 (0 或 1)
        game (GeneralSumGame): 為two-player approximation, 其payoff matrix的shape為(4, 4, 2)
        strategy (list): 玩家策略的機率, 長度為8

    Returns:
        float: 指定玩家的期望payoff
    """
    # 分離玩家的策略
    n_players = game.n_players
    player_strategy = np.reshape(strategy, (n_players, -1))
    
    # 初始化期望payoff
    player_expected_payoff = np.zeros(n_players)

    # 遍歷所有可能的動作組合
    all_actions = list(product([0, 1], repeat=n_players))
    for joint_action in all_actions:
        # 計算這個動作組合的聯合機率
        probability = 1.0
        for i in range(n_players):
            probability *= player_strategy[i, joint_action[i]]
        
        # 獲取對應的payoff並加權相加
        player_expected_payoff += probability * game.payoff_matrix[joint_action]

    if player_idx is None:
        return player_expected_payoff
    return player_expected_payoff[player_idx]
    
def feasibility_run(game:GeneralSumGame, n_players=4, tol=1e-8, init_val=None) -> np.ndarray:  # 設置容忍值:
    n_players = n_players
    #two_player_approximation = game.two_virtual_player_approximation()
    # 1. 邊界
    bounds = Bounds(0.0, 1.0)
    player_strategy_probs = []
    # b = np.array([1.0])
    # A = np.ones((1, 6))#np.array([[0.0, 1.0, 1.0, 1.0]]) # 0 <= g0+g1+g2 <= 1
    # A[0, 0] = 0.0
    # A[0, 1] = 0.0
    # A[0, 2] = 0.0
    # player_strategy_probs.append(LinearConstraint(A, 0.0, b))
    # A = np.ones((1, 6))#np.array([[0.0, 1.0, 1.0, 1.0]]) # 0 <= h0+h1+h2 <= 1
    # A[0, 3] = 0.0
    # A[0, 4] = 0.0
    # A[0, 5] = 0.0
    # player_strategy_probs.append(LinearConstraint(A, 0.0, b))

    for i in range(2):
        A = np.zeros((1, 2))
        A[0, i] = 1.0
        player_strategy_probs.append(LinearConstraint(A, 0.0, 1.0))

    # 2. 定義變數相關的非線性約束
    def best_reponse(x):
        """
        v1(k)+r_{1,k} = U*1, k=0,1
        v2(l)+r_{2,l} = U*2, l=0,1,...,n-1
        """
        all_strategy = get_all_strategy(x)
        all_payoffs = calculate_expected_payoff(None, game, all_strategy)
        u1_star, u2_star = all_payoffs[0], all_payoffs[-1]
        # Team 1
        v1_0 = np.concatenate((np.array([1.0, 0.0], dtype=np.float64), all_strategy[2:]))
        v1_1 = np.concatenate((np.array([0.0, 1.0], dtype=np.float64), all_strategy[2:]))

        v1_0 = calculate_expected_payoff(0, game, v1_0)
        v1_1 = calculate_expected_payoff(0, game, v1_1)

        
        # Team 2
        v2_0 = np.concatenate((all_strategy[:6], np.array([1.0, 0.0], dtype=np.float64)))
        v2_1 = np.concatenate((all_strategy[:6], np.array([0.0, 1.0], dtype=np.float64)))

        v2_0 = calculate_expected_payoff(-1, game, v2_0)
        v2_1 = calculate_expected_payoff(-1, game, v2_1)

        r = np.array([u1_star, u1_star, u2_star, u2_star]) - \
            np.array([v1_0, v1_1, v2_0, v2_1])

        indifference = r *  np.concatenate((all_strategy[:2], all_strategy[6:]))

        #all_strategy = all_strategy#.round(3)
        #g0, g1, g2, g3 = all_strategy[2:]

        return np.concatenate((r, indifference))#, np.array([g0*g3 - g2*g1])))
    

    # 非線性約束
    n_r = 4  # r 的元素個數
    n_indifference = 4  # indifference 的元素個數
    lower_bounds = np.concatenate((np.zeros(n_r), np.zeros(n_indifference)))  # r 的下界為 0, indifference 的下界為 0
    upper_bounds = np.concatenate((np.inf * np.ones(n_r), np.zeros(n_indifference)))  # r 的上界為 0, indifference 的上界為 0
    best_reponse_constraint = NonlinearConstraint(best_reponse, lower_bounds, upper_bounds)  # v1(k)+r_{1,k} - U*1 = 0、r>=0 and x*r=0
    #payoff_ge_0_constraint = NonlinearConstraint(payoff_ge_0, 0, np.inf)  # >= 0
    #indifference_constraint = NonlinearConstraint(indifference, 0, 0)  # = 0

    # 3. 定義線性約束
    # A = np.array([[1, 1, 0]])  # x1 + x2 <= 1
    # b = np.array([1])
    # linear_constraint = LinearConstraint(A, -np.inf, b)

    # 4.虛擬目標函數
    def dummy_objective(x):
        return 0  # 沒有實際目標函數
    
    
    def basic_penlity(x):
        sum = 0
        strategys = get_all_strategy(x)
        for p in strategys:
            if p < 0.0 or p > 1.0:
                sum += 1
        return sum
    
    def best_response_penlity(x):
        sum = 0
        indf = best_reponse(x)
        for i in range(n_r + n_indifference):
            #if indf[i] < lower_bounds[i] or indf[i] > upper_bounds[i]:
            if (not np.isclose(indf[i], lower_bounds[i], atol=1e-4)) or (not np.isclose(indf[i], upper_bounds[i], atol=1e-4)):
                sum += 1
        return sum

    def penality(x):
        return dummy_objective(x) + best_response_penlity(x) + basic_penlity(x) *10

    # 5.設定變數初始值
    if init_val is not None:
        strategys = init_val # 有2個變數, 分別為[x0, y0], 表示symmetric strategy
    else:
        strategys = [0.5, 0.5]

    # 6.求解
    constraints=[
                #indepent_prob_constraint,
                best_reponse_constraint,
                #payoff_ge_0_constraint,
                #indifference_constraint,
                ] + player_strategy_probs

    result = minimize(
        dummy_objective,
        strategys,
        bounds=bounds,  # 設置變數邊界
        constraints=constraints,
        method='trust-constr',
        #method='COBYLA',
        tol=tol,  # 設置容忍值
        options={'factorization_method': 'SVDFactorization',# 'QRFactorization' 'SVDFactorization'
                 'xtol': tol,  # 解的容忍值
                 'barrier_tol': tol,  # 障礙函數的容忍值
                 'maxiter': 3000 if init_val is None else 0,  # 最大迭代次數
                }
    )

    # print("Constraint Status:", result.success)
    strategys = get_all_strategy(result.x.round(3))#np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])#
    basic_prob_cons = basic_penlity(result.x.round(2))
    best_rep_cons = best_response_penlity(result.x.round(2))
    # print("Basic Probability Constraint:", basic_prob_cons)
    # print("Best Response Constraint:", best_rep_cons)

    # Record 2-player approximate game constraint infomation
    info = {}
    all_strategy = get_all_strategy(result.x)#strategys.copy()#
    expected_payoff = calculate_expected_payoff(None, game, all_strategy)
    u1_star, u2_star = expected_payoff[0], expected_payoff[-1]
    # Player 1
    v1_0 = np.concatenate((np.array([1.0, 0.0], dtype=np.float64), all_strategy[2:]))
    v1_1 = np.concatenate((np.array([0.0, 1.0], dtype=np.float64), all_strategy[2:]))
    
    v1_0 = calculate_expected_payoff(0, game, v1_0)
    v1_1 = calculate_expected_payoff(0, game, v1_1)
    
    # Virtual Player
    v2_0 = np.concatenate((all_strategy[:6], np.array([1.0, 0.0], dtype=np.float64)))
    v2_1 = np.concatenate((all_strategy[:6], np.array([0.0, 1.0], dtype=np.float64)))
    
    v2_0 = calculate_expected_payoff(3, game, v2_0)
    v2_1 = calculate_expected_payoff(3, game, v2_1)

    r = np.array([u1_star, u1_star, u2_star, u2_star]) - \
        np.array([v1_0, v1_1, v2_0, v2_1])

    indifference = r * np.concatenate((all_strategy[:2], all_strategy[6:]))

    info['u'] = np.array([u1_star, u1_star, u2_star, u2_star])
    info['v'] = np.array([v1_0, v1_1, v2_0, v2_1])
    info['r'] = r
    info['indifference'] = indifference
    info['constraint'] = basic_prob_cons + best_rep_cons
    
    return strategys.round(4), result, info

if __name__ == "__main__":
    # 產生Game
    i = 0
    n_players = 4
    tol=1e-8  # 設置容忍值
    game = TeamGame(n_players)
    two_player_approximation = game.two_player_approximation(i)
    strategy, result, info = feasibility_run(game, tol=tol)
    print(result)
    print(strategy)
    