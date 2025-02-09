import cvxpy as cp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from game_generator import GeneralSumGame
from judger import NashEquilibriumJudger
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
    # Virtual Player 1
    h0, h1, h2 = x[0], x[1], x[2]
    h3 = 1.0 - h0 - h1 - h2
    # Virtual Player 2
    g0, g1, g2 = x[3], x[4], x[5]
    g3 = 1.0 - g0 - g1 - g2
    return np.array([h0, h1, h2, h3, g0, g1, g2, g3])

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
    player_0_strategy = strategy[:4]
    player_1_strategy = strategy[4:]
    
    # 初始化期望payoff
    player_0_expected_payoff = 0.0
    player_1_expected_payoff = 0.0

    # 遍歷所有可能的動作組合
    for action_0 in range(4):  # Player 1 has 4 actions
        for action_1 in range(4):  # Player 2 has 4 actions
            # 計算這個動作組合的聯合機率
            probability = player_0_strategy[action_0] * player_1_strategy[action_1]
            
            # 獲取對應的payoff
            player_0_payoff = game.payoff_matrix[action_0, action_1, 0]
            player_1_payoff = game.payoff_matrix[action_0, action_1, 1]
            
            # 加入到各自的期望值
            player_0_expected_payoff += probability * player_0_payoff
            player_1_expected_payoff += probability * player_1_payoff

    if player_idx == 0:
        return player_0_expected_payoff
    elif player_idx == 1:
        return player_1_expected_payoff
    elif player_idx is None:
        return player_0_expected_payoff, player_1_expected_payoff
    else:
        raise ValueError("player_idx 必須是 0, 1 或 None")
    
def feasibility_run(game:GeneralSumGame, n_players=4, tol=1e-8, independent=True) -> np.ndarray:  # 設置容忍值:
    n_players = n_players
    two_player_approximation = game.two_virtual_player_approximation()
    # 1. 邊界
    bounds = Bounds(0.0, 1.0)
    player_strategy_probs = []
    b = np.array([1.0])
    A = np.ones((1, 6))#np.array([[0.0, 1.0, 1.0, 1.0]]) # 0 <= g0+g1+g2 <= 1
    A[0, 0] = 0.0
    A[0, 1] = 0.0
    A[0, 2] = 0.0
    player_strategy_probs.append(LinearConstraint(A, 0.0, b))
    A = np.ones((1, 6))#np.array([[0.0, 1.0, 1.0, 1.0]]) # 0 <= h0+h1+h2 <= 1
    A[0, 3] = 0.0
    A[0, 4] = 0.0
    A[0, 5] = 0.0
    player_strategy_probs.append(LinearConstraint(A, 0.0, b))

    for i in range(6):
        A = np.zeros((1, 6))#np.array([[1.0, 0.0, 0.0, 0.0]]) # x
        A[0, i] = 1.0
        player_strategy_probs.append(LinearConstraint(A, 0.0, b))

    # 2. 定義變數相關的非線性約束
    def best_reponse(x):
        """
        v1(k)+r_{1,k} = U*1, k=0,1
        v2(l)+r_{2,l} = U*2, l=0,1,...,n-1
        """
        all_strategy = get_all_strategy(x)
        u1_star, u2_star = calculate_expected_payoff(None, two_player_approximation, all_strategy)
        # # Player 1
        v1_0 = np.concatenate((np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), all_strategy[4:]))
        v1_1 = np.concatenate((np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64), all_strategy[4:]))
        v1_2 = np.concatenate((np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64), all_strategy[4:]))
        v1_3 = np.concatenate((np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), all_strategy[4:]))
        v1_0 = calculate_expected_payoff(0, two_player_approximation, v1_0)
        v1_1 = calculate_expected_payoff(0, two_player_approximation, v1_1)
        v1_2 = calculate_expected_payoff(0, two_player_approximation, v1_2)
        v1_3 = calculate_expected_payoff(0, two_player_approximation, v1_3)
        
        # Virtual Player
        v2_0 = np.concatenate((all_strategy[:4], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)))
        v2_1 = np.concatenate((all_strategy[:4], np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)))
        v2_2 = np.concatenate((all_strategy[:4], np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)))
        v2_3 = np.concatenate((all_strategy[:4], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))
        v2_0 = calculate_expected_payoff(1, two_player_approximation, v2_0)
        v2_1 = calculate_expected_payoff(1, two_player_approximation, v2_1)
        v2_2 = calculate_expected_payoff(1, two_player_approximation, v2_2)
        v2_3 = calculate_expected_payoff(1, two_player_approximation, v2_3)

        r = np.array([u1_star, u1_star, u1_star, u1_star, u2_star, u2_star, u2_star, u2_star]) - \
            np.array([v1_0, v1_1, v1_2, v1_3, v2_0, v2_1, v2_2, v2_3])

        indifference = r * all_strategy

        #all_strategy = all_strategy#.round(3)
        #g0, g1, g2, g3 = all_strategy[2:]

        return np.concatenate((r, indifference))#, np.array([g0*g3 - g2*g1])))
    
    def indepent_prob(x):
        # g0 * g3 = g1 * g2
        g0, g1, g2 = x[3], x[4], x[5]
        g3 = 1.0 - g0 - g1 - g2

        return g0*g3 - g2*g1
    
    def indepent_prob2(x):
        # g0 * g3 = g1 * g2
        h0, h1, h2 = x[0], x[1], x[2]
        h3 = 1.0 - h0 - h1 - h2

        return h0*h3 - h2*h1
        # g0, g1, g2 = x[1], x[2], x[3]
        # g3 = 1.0 - g0 - g1 - g2
        # px0 = g0 + g2  # P(X=0)
        # px1 = g1 + g3  # P(X=1)
        # py0 = g0 + g1  # P(Y=0)
        # py1 = g2 + g3  # P(Y=1)

        # # 檢查每個條件
        # constraints = []
        # if not np.isclose(g0, 0, 1e-8):
        #     constraints.append(g0 - px0 * py0)  # g0 = P(X=0) * P(Y=0)
        # if not np.isclose(g1, 0, 1e-8):
        #     constraints.append(g1 - px1 * py0)  # g1 = P(X=1) * P(Y=0)
        # if not np.isclose(g2, 0, 1e-8):
        #     constraints.append(g2 - px0 * py1)  # g2 = P(X=0) * P(Y=1)
        # if not np.isclose(g3, 0, 1e-8):
        #     constraints.append(g3 - px1 * py1)  # g3 = P(X=1) * P(Y=1)

        # return np.sum(np.abs(constraints))  # 返回所有約束的絕對值和

    # 非線性約束
    n_r = 8  # r 的元素個數
    n_indifference = 8  # indifference 的元素個數
    lower_bounds = np.concatenate((np.zeros(n_r), np.zeros(n_indifference)))  # r 的下界為 0, indifference 的下界為 0
    upper_bounds = np.concatenate((np.inf * np.ones(n_r), np.zeros(n_indifference)))  # r 的上界為 0, indifference 的上界為 0
    best_reponse_constraint = NonlinearConstraint(best_reponse, lower_bounds, upper_bounds)  # v1(k)+r_{1,k} - U*1 = 0、r>=0 and x*r=0
    #payoff_ge_0_constraint = NonlinearConstraint(payoff_ge_0, 0, np.inf)  # >= 0
    indepent_prob_constraint = NonlinearConstraint(indepent_prob, 0.0, 0.0 )  # g0 * g3 - g1 * g2 = 0
    indepent_prob_constraint2 = NonlinearConstraint(indepent_prob2, 0.0, 0.0 )  # h0 * h3 - h1 * h2 = 0
    #indifference_constraint = NonlinearConstraint(indifference, 0, 0)  # = 0

    # 3. 定義線性約束
    # A = np.array([[1, 1, 0]])  # x1 + x2 <= 1
    # b = np.array([1])
    # linear_constraint = LinearConstraint(A, -np.inf, b)

    # 4.虛擬目標函數
    def dummy_objective(x):
        return 0  # 沒有實際目標函數
    
    def independent_penlity(x):
        if independent and not np.isclose(indepent_prob(x), 0.0):
            return 4
        return 0
    
    def basic_penlity(x):
        sum = 0
        strategys = get_all_strategy(x)
        for p in strategys:
            if p < 0.0 or p > 1.0:
                sum += 1
        if not np.isclose(np.sum(strategys[4:]), 1.0):
            sum += 3
        if not np.isclose(np.sum(strategys[:4]), 1.0):
            sum += 3            
        return sum
    
    def best_response_penlity(x):
        sum = 0
        indf = best_reponse(x)
        for i in range(n_r):
            if indf[i] < lower_bounds[i] or indf[i] > upper_bounds[i]:
                sum += 1
        return sum

    def penality(x):
        return dummy_objective(x) + (best_response_penlity(x) + independent_penlity(x)) + basic_penlity(x) *10

    # 5.設定變數初始值
    strategys = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25] # 有6個變數，分別為h0, h1, h2, g0、g1、g2

    # 6.求解
    constraints=[
                #indepent_prob_constraint,
                best_reponse_constraint,
                #payoff_ge_0_constraint,
                #indifference_constraint,
                ] + player_strategy_probs
    
    if independent:
        constraints.insert(0, indepent_prob_constraint)
        constraints.insert(0, indepent_prob_constraint2)

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
                 'maxiter': 3000,  # 最大迭代次數
                 
                }
    )

    # print("Constraint Status:", result.success)
    strategys = get_all_strategy(result.x.round(3))#np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])#
    print("Basic Probability Constraint:", basic_penlity(result.x.round(2)))
    print("Best Response Constraint:", best_response_penlity(result.x.round(2)))#result.x.round(2)))
    if independent:
        print("Independent Constraint:", independent_penlity(result.x.round(2)))
    # print("Solution:", strategys.round(3))
    # print("Solution:", result.x)

    # Record 2-player approximate game constraint infomation
    info = {}
    all_strategy = get_all_strategy(result.x.round(5))#strategys.copy()#
    u1_star, u2_star = calculate_expected_payoff(None, two_player_approximation, all_strategy)
    # Player 1
    v1_0 = np.concatenate((np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), all_strategy[4:]))
    v1_1 = np.concatenate((np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64), all_strategy[4:]))
    v1_2 = np.concatenate((np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64), all_strategy[4:]))
    v1_3 = np.concatenate((np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), all_strategy[4:]))
    v1_0 = calculate_expected_payoff(0, two_player_approximation, v1_0)
    v1_1 = calculate_expected_payoff(0, two_player_approximation, v1_1)
    v1_2 = calculate_expected_payoff(0, two_player_approximation, v1_2)
    v1_3 = calculate_expected_payoff(0, two_player_approximation, v1_3)
    
    # Virtual Player
    v2_0 = np.concatenate((all_strategy[:4], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)))
    v2_1 = np.concatenate((all_strategy[:4], np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)))
    v2_2 = np.concatenate((all_strategy[:4], np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)))
    v2_3 = np.concatenate((all_strategy[:4], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))
    v2_0 = calculate_expected_payoff(1, two_player_approximation, v2_0)
    v2_1 = calculate_expected_payoff(1, two_player_approximation, v2_1)
    v2_2 = calculate_expected_payoff(1, two_player_approximation, v2_2)
    v2_3 = calculate_expected_payoff(1, two_player_approximation, v2_3)

    r = np.array([u1_star, u1_star, u1_star, u1_star, u2_star, u2_star, u2_star, u2_star]) - \
        np.array([v1_0, v1_1, v1_2, v1_3, v2_0, v2_1, v2_2, v2_3])

    indifference = r * all_strategy

    info['u'] = np.array([u1_star, u1_star, u1_star, u1_star, u2_star, u2_star, u2_star, u2_star])
    info['v'] = np.array([v1_0, v1_1, v1_2, v1_3, v2_0, v2_1, v2_2, v2_3])
    info['r'] = r
    info['indifference'] = indifference
    if independent:
        info['independent'] = all_strategy[0]*all_strategy[3]-all_strategy[1]*all_strategy[2]
        info['independent2'] = all_strategy[4]*all_strategy[7]-all_strategy[5]*all_strategy[6]
    
    return strategys.round(3), result, info

if __name__ == "__main__":
    # 產生Game
    i = 0
    n_players = 3
    tol=1e-8  # 設置容忍值
    game = GeneralSumGame(n_players)
    two_player_approximation = game.two_player_approximation(i)
    strategy, result, info = feasibility_run(game, independent=False, tol=tol)
    print(result)
    print(strategy)
    exit()
    # 1. 邊界
    bounds = Bounds(0.0, 1.0)
    player_strategy_probs = []
    A = np.array([[0.0, 1.0, 1.0, 1.0]]) # g0+g1+g2 <= 1
    b = np.array([1.0])
    player_strategy_probs.append(LinearConstraint(A, 0.0, b))

    for i in range(1 + 2**(n_players-1) - 1):
        A = np.zeros((1, 4))
        A[0, i] = 1.0
        player_strategy_probs.append(LinearConstraint(A, 0.0, b))

    # 2. 定義變數相關的非線性約束
    def best_reponse(x):
        """
        v1(k)+r_{1,k} = U*1, k=0,1
        v2(l)+r_{2,l} = U*2, l=0,1,...,n-1
        """
        all_strategy = get_all_strategy(x)
        u1_star, u2_star = calculate_expected_payoff(None, two_player_approximation, all_strategy)
        # # Player 1
        v1_0 = np.concatenate((np.array([1.0, 0.0], dtype=np.float64), all_strategy[2:]))
        v1_1 = np.concatenate((np.array([0.0, 1.0], dtype=np.float64), all_strategy[2:]))
        v1_0 = calculate_expected_payoff(0, two_player_approximation, v1_0)
        v1_1 = calculate_expected_payoff(0, two_player_approximation, v1_1)
        
        # Virtual Player
        v2_0 = np.concatenate((all_strategy[:2], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)))
        v2_1 = np.concatenate((all_strategy[:2], np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)))
        v2_2 = np.concatenate((all_strategy[:2], np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)))
        v2_3 = np.concatenate((all_strategy[:2], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))
        v2_0 = calculate_expected_payoff(1, two_player_approximation, v2_0)
        v2_1 = calculate_expected_payoff(1, two_player_approximation, v2_1)
        v2_2 = calculate_expected_payoff(1, two_player_approximation, v2_2)
        v2_3 = calculate_expected_payoff(1, two_player_approximation, v2_3)

        r = np.array([u1_star, u1_star, u2_star, u2_star, u2_star, u2_star]) - \
            np.array([v1_0, v1_1, v2_0, v2_1, v2_2, v2_3])

        indifference = r * all_strategy

        return np.concatenate((r, indifference))

    def payoff_ge_0(x):
        return np.sin(x[2]) - 0.5  # 舉例變數相關約束: sin(x3) <= 0.5
    
    def indepent_prob(x):
        # g0 * g3 = g1 * g2
        g0, g1, g2 = x[1], x[2], x[3]
        g3 = 1.0 - g0 - g1 - g2

        return g0*g3 - g2*g1
        # g0, g1, g2 = x[1], x[2], x[3]
        # g3 = 1.0 - g0 - g1 - g2
        # px0 = g0 + g2  # P(X=0)
        # px1 = g1 + g3  # P(X=1)
        # py0 = g0 + g1  # P(Y=0)
        # py1 = g2 + g3  # P(Y=1)

        # # 檢查每個條件
        # constraints = []
        # if not np.isclose(g0, 0, 1e-8):
        #     constraints.append(g0 - px0 * py0)  # g0 = P(X=0) * P(Y=0)
        # if not np.isclose(g1, 0, 1e-8):
        #     constraints.append(g1 - px1 * py0)  # g1 = P(X=1) * P(Y=0)
        # if not np.isclose(g2, 0, 1e-8):
        #     constraints.append(g2 - px0 * py1)  # g2 = P(X=0) * P(Y=1)
        # if not np.isclose(g3, 0, 1e-8):
        #     constraints.append(g3 - px1 * py1)  # g3 = P(X=1) * P(Y=1)

        # return np.sum(np.abs(constraints))  # 返回所有約束的絕對值和

    # 非線性約束
    n_r = 6  # r 的元素個數
    n_indifference = 6  # indifference 的元素個數
    lower_bounds = np.concatenate((np.zeros(n_r, dtype=int), np.zeros(n_indifference, dtype=int)))  # r 的下界為 0, indifference 的下界為 0
    upper_bounds = np.concatenate((np.inf * np.ones(n_r), np.zeros(n_indifference)))  # r 的上界為 0, indifference 的上界為 0
    best_reponse_constraint = NonlinearConstraint(best_reponse, lower_bounds, upper_bounds)  # v1(k)+r_{1,k} - U*1 = 0、r>=0 and x*r=0
    #payoff_ge_0_constraint = NonlinearConstraint(payoff_ge_0, 0, np.inf)  # >= 0
    indepent_prob_constraint = NonlinearConstraint(indepent_prob, 0.0, 0.0)  # g0 * g3 - g1 * g2 = 0
    #indifference_constraint = NonlinearConstraint(indifference, 0, 0)  # = 0

    # 3. 定義線性約束
    # A = np.array([[1, 1, 0]])  # x1 + x2 <= 1
    # b = np.array([1])
    # linear_constraint = LinearConstraint(A, -np.inf, b)

    # 4.虛擬目標函數
    def dummy_objective(x):
        return 0  # 沒有實際目標函數

    # 5.設定變數初始值
    strategys = [0.5, 0.25, 0.25, 0.25] # 有4個變數，分別為x, g0、g1、g2

    # 6.求解
    constraints=[
                indepent_prob_constraint,
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
        tol=tol,  # 設置容忍值
        options={'factorization_method': 'SVDFactorization',
                 'xtol': 1e-8,  # 解的容忍值
                 'barrier_tol': 1e-8,  # 障礙函數的容忍值
                 'maxiter': 10000  # 最大迭代次數
                }
    )

    print("Constraint Status:", result.success)
    strategys = get_all_strategy(result.x.round(2))
    print("Solution:", strategys.round(3).reshape((-1, 2)))
    print("Solution:", result.x)
