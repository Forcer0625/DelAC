import pygambit as gbt
from judger import *
from game_generator import *
from feasibility4 import *
import csv

def normal2gamebit(payoff_matrix):
    pass

def gamebit2normal(payoff_matrix):
    pass

def game_transform(payoff_matrix, player_idx):
    pass

def algo_run(game:GeneralSumGame, save_id=None) -> np.ndarray:
    n_players = game.n_players
    strategy = np.zeros((n_players, 2))
    oppo_player_strategy = []
    for i in range(n_players):
        i_th_two_player_game = game.two_player_approximation(i)
        result = gbt.nash.lcp_solve(i_th_two_player_game.gamebit_form, rational=False, stop_after=1).equilibria
        i_th_two_player_strategy = []    
        #print(result[0])
        for act in result[0]['Player 1']:
            i_th_two_player_strategy.append(act[1])
        i_th_two_player_strategy = np.array(i_th_two_player_strategy)
        strategy[i] = i_th_two_player_strategy

        # 紀錄Two-Player Game中，Player 2的Nash Equilibria
        if save_id is not None and game.n_players == 3:
            temp = []
            for act in result[0]['Player 2']:
                temp.append(act[1])
            oppo_player_strategy.append(np.array(temp))
    
    if save_id is not None and game.n_players == 3:
        expected_payoff = NashEquilibriumJudger.get_payoff(strategy, game.payoff_matrix)
        n_steps = 10000
        epsilon = 0.01
        is_nash = NashEquilibriumJudger.run(strategy, game.payoff_matrix, n_steps=n_steps, epsilon=epsilon)
        with open('./runs/'+save_id+'.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)

            for player3_action in [0, 1]:
                writer.writerow(['Player 3: '+str(player3_action), 'Player 2: 0', 'Player 2: 1'])
                for player1_action in [0, 1]:
                    write_data = ['Player 1: '+str(player1_action)]
                    #pay_off_str = '('
                    for player2_acton in [0, 1]:
                        joint_action = (player1_action, player2_acton, player3_action)
                        pay_off_str = ', '.join(str(x) for x in game.payoff_matrix[joint_action])
                        #pay_off_str += ')'
                        write_data.append(pay_off_str)
                    writer.writerow(write_data)
                writer.writerow([])
            
            # 寫入每個i-th two player game的結果
            for i in range(game.n_players):
                writer.writerow(['Player ' + str(i+1) + '\'s Two-Player Game'])
                player1_strategy_str = '[' + ', '.join(str(round(x, 2)) for x in strategy[i]) + ']'
                player2_strategy_str = '[' + ', '.join(str(round(x, 2)) for x in oppo_player_strategy[i]) + ']'
                writer.writerow([player1_strategy_str, player2_strategy_str])
                # Test Joint Probability -> PMF
                player_pmfs = compute_marginal_pmf(oppo_player_strategy[i])
                for j in range(n_players - 1):
                    player_pmf = '[' + ', '.join(str(round(x, 2)) for x in player_pmfs[j]) + ']'
                    writer.writerow([player_pmf])
                player_pmfs.insert(i, strategy[i])
                player_pmfs = np.array(player_pmfs)
                mariginal_is_nash = NashEquilibriumJudger.run(player_pmfs, game.payoff_matrix, n_steps=n_steps, epsilon=epsilon)
                writer.writerow([str(mariginal_is_nash[0])])
                writer.writerow([])

            # 寫入相關資訊
            writer.writerow(['Expected Payoff'])
            writer.writerow([round(x, 2) for x in expected_payoff])
            writer.writerow([])
            writer.writerow(['Is a Nash?', 'Eval.Steps', 'Epsilon'])
            writer.writerow([str(is_nash[0]), n_steps, str(epsilon*100)+'%'])
            if not is_nash[0]:
                writer.writerow(['Better strategy for ' + is_nash[1]['Player'],\
                                 'New Payoff for ' + is_nash[1]['Player']])
                better_strategy = is_nash[1][is_nash[1]['Player']]
                better_strategy = '[' + ', '.join(str(round(x, 2)) for x in better_strategy) + ']'
                writer.writerow([better_strategy, round(is_nash[1]['New Payoff'], 2)])

    return strategy

def get_equilibria(result) -> np.ndarray:
    strategy = []
    oppo_strategy = []
    for act in result[0]['Player 1']:
        strategy.append(act[1])
    for act in result[0]['Player 2']:
        oppo_strategy.append(act[1])
    return np.array(strategy), np.array(oppo_strategy)

def solve_feasibility_program(i, pi, previous_g):
    pass

def compute_next_g(previous_g):
    pass

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

def algo_v2(game:GeneralSumGame):
    n_players = game.n_players
    strategy = np.zeros((n_players, 2))
    g = []

    # i = 0, first player
    two_player_game = game.two_player_approximation(0)
    result = gbt.nash.lcp_solve(two_player_game.gamebit_form, rational=False, stop_after=1).equilibria
    
    strategy[0], g1 = get_equilibria(result)
    g.append(g1)
    # i = 1...n-1 player, iteratively compute feasibility program
    for i in range(1, n_players):
        pi_i = solve_feasibility_program(i, strategy, g[i-1])
        next_g = compute_next_g(g[i-1])
        strategy[i] = pi_i
        g.append(next_g)

    return strategy

def algo_run_v4(game:GeneralSumGame, save_id=None) -> np.ndarray:
    n_players = game.n_players
    gbt_strategy = np.zeros((n_players, 2))
    #strategy = np.zeros((n_players, 2))
    
    i_th_two_player_game = game.two_virtual_player_approximation() # Two virtual player
    feasibility_strategy, _, feasibility_info = feasibility_run(game)
    feasibility_strategy = feasibility_strategy.reshape(n_players, 2)
    joint_dist1 = np.concatenate((feasibility_strategy[0], feasibility_strategy[1]))
    mariginal_strategy1 = compute_marginal_pmf(joint_dist1)
    joint_dist2 = np.concatenate((feasibility_strategy[2], feasibility_strategy[3]))
    mariginal_strategy2 = compute_marginal_pmf(joint_dist2)
    feasibility_strategy[0], feasibility_strategy[1] = mariginal_strategy1[0], mariginal_strategy1[1]
    feasibility_strategy[2], feasibility_strategy[3] = mariginal_strategy2[0], mariginal_strategy2[1]
    gbt_result = gbt.nash.ipa_solve(game.gamebit_form).equilibria

    # without independent constraint
    # feasibility_strategy_woindcon, _, feasibility_woindcon_info = feasibility_run(game, independent=False)
    # feasibility_strategy_woindcon = feasibility_strategy_woindcon.reshape(3, 2)
    # joint_dist_woindcon = np.concatenate((feasibility_strategy_woindcon[1], feasibility_strategy_woindcon[2]))
    # mariginal_strategy_woindcon = compute_marginal_pmf(joint_dist_woindcon)
    # feasibility_strategy_woindcon[1], feasibility_strategy_woindcon[2] = mariginal_strategy_woindcon[0], mariginal_strategy_woindcon[1]
    
    # Get gbt strategy
    for p in range(n_players):
        i_th_player_strategy = []
        for act in gbt_result[0]['Player '+str(p+1)]:
            i_th_player_strategy.append(act[1])
        gbt_strategy[p] = np.array(i_th_player_strategy)

    # 紀錄Two-Player Game中，Player 2的Nash Equilibria
    if save_id is not None and game.n_players == 4:
    
        with open('./runs/'+save_id+'.csv', 'w', newline='') as csvfile:
            # 建立 CSV 檔寫入器
            writer = csv.writer(csvfile)
            for player4_action in [0, 1]:
                writer.writerow(['Player 4: '+str(player4_action)])
                for player3_action in [0, 1]:
                    writer.writerow(['Player 3: '+str(player3_action), 'Player 2: 0', 'Player 2: 1'])
                    for player1_action in [0, 1]:
                        write_data = ['Player 1: '+str(player1_action)]
                        #pay_off_str = '('
                        for player2_acton in [0, 1]:
                            joint_action = (player1_action, player2_acton, player3_action, player4_action)
                            pay_off_str = ', '.join(str(x) for x in game.payoff_matrix[joint_action])
                            #pay_off_str += ')'
                            write_data.append(pay_off_str)
                        writer.writerow(write_data)
                    writer.writerow([])
                writer.writerow([])

            
            writer.writerow(['Two Vritual Player Game'])
            strategy = [gbt_strategy, feasibility_strategy]
            algo_str = ['Gamebit Result', 'Feasibility Result']
            for algo in range(2):
                writer.writerow([])
                expected_payoff = NashEquilibriumJudger.get_payoff(strategy[algo], game.payoff_matrix)
                n_steps = 10000
                epsilon = 0.01
                is_nash = NashEquilibriumJudger.run(strategy[algo], game.payoff_matrix, n_steps=n_steps, epsilon=epsilon)
            
                writer.writerow([algo_str[algo]])
                player_strategy_str = []
                for i in range(n_players):
                    player_strategy_str.append('[' + ', '.join(str(round(x, 2)) for x in strategy[algo][i]) + ']')
                writer.writerow(player_strategy_str)

                if algo >= 1:
                    # # h0~h3
                    # player_strategy.append(joint_dist1)
                    # player_strategy = [strategy[algo][0]]
                    # player_strategy.append(joint_dist2)
                    # is_nash_woindcon = NashEquilibriumJudger.run_indifference(player_strategy, i_th_two_player_game.payoff_matrix)
                    # is_nash_write_data = ['[' + ', '.join(str(round(x, 2)) for x in (joint_dist2)) + ']', \
                    #                  str(is_nash_woindcon[0])]
                    
                    # if not is_nash_woindcon[0]:
                    #     writer.writerow(['h0, h1, h2, h3', 'Is a Nash?', 'Player', 'Strategy', 'Old Payoff', 'New Payoff'])
                    #     is_nash_write_data.append(is_nash_woindcon[1]['Player'])
                    #     is_nash_write_data.append('[' + ', '.join(str(round(x, 2)) for x in is_nash_woindcon[1]['better strategy']) + ']')
                    #     is_nash_write_data.append(str(is_nash_woindcon[1]['Old Payoff']))
                    #     is_nash_write_data.append(str(is_nash_woindcon[1]['New Payoff']))
                    # else:
                    #     writer.writerow(['h0, h1, h2, h3', 'Is a Nash?'])

                    # writer.writerow(is_nash_write_data)
                    # # g0~g3
                    # player_strategy = [strategy[algo][0], strategy[algo][1]]
                    # player_strategy.append(joint_dist2)
                    # is_nash_woindcon = NashEquilibriumJudger.run_indifference(player_strategy, i_th_two_player_game.payoff_matrix)
                    # is_nash_write_data = ['[' + ', '.join(str(round(x, 2)) for x in (joint_dist2)) + ']', \
                    #                  str(is_nash_woindcon[0])]
                    
                    # if not is_nash_woindcon[0]:
                    #     writer.writerow(['g0, g1, g2, g3', 'Is a Nash?', 'Player', 'Strategy', 'Old Payoff', 'New Payoff'])
                    #     is_nash_write_data.append(is_nash_woindcon[1]['Player'])
                    #     is_nash_write_data.append('[' + ', '.join(str(round(x, 2)) for x in is_nash_woindcon[1]['better strategy']) + ']')
                    #     is_nash_write_data.append(str(is_nash_woindcon[1]['Old Payoff']))
                    #     is_nash_write_data.append(str(is_nash_woindcon[1]['New Payoff']))
                    # else:
                    #     writer.writerow(['g0, g1, g2, g3', 'Is a Nash?'])

                    # writer.writerow(is_nash_write_data)
                    
                    # Constraint Info
                    info = feasibility_info
                    writer.writerow([])
                    writer.writerow(['', '1,0', '1,1', '1,2', '1,3', '2,0', '2,1', '2,2', '2,3'])
                    all_cons = ['u', 'v', 'r', 'indifference']
                    for cons in all_cons:    
                        constraint_data = [cons]
                        for x in info[cons]:
                            constraint_data.append(str(x))
                        writer.writerow(constraint_data)
                        
                    writer.writerow([])

                    # Original game constraint
                    # writer.writerow(['', '1,0', '1,1', '2,0', '2,1', '3,0', '3,1', '4,0', '4,1'])
                    # all_cons = ['u', 'v', 'r', 'indifference']
                    # for cons in all_cons:    
                    #     constraint_data = [cons]
                    #     for x in info['_' + cons]:
                    #         constraint_data.append(str(x))
                    #     writer.writerow(constraint_data)
                        
                    # writer.writerow([])

                # 寫入相關資訊
                writer.writerow(['Expected Payoff'])
                writer.writerow([round(x, 2) for x in expected_payoff])
                writer.writerow([])
                writer.writerow(['Is a Nash?', 'Eval.Steps', 'Epsilon'])
                writer.writerow([str(is_nash[0]), n_steps, str(epsilon*100)+'%'])
                if not is_nash[0]:
                    writer.writerow(['Better strategy for ' + is_nash[1]['Player'],\
                                    'New Payoff for ' + is_nash[1]['Player']])
                    better_strategy = is_nash[1][is_nash[1]['Player']]
                    better_strategy = '[' + ', '.join(str(round(x, 2)) for x in better_strategy) + ']'
                    writer.writerow([better_strategy, round(is_nash[1]['New Payoff'], 2)])
                    
    return strategy


if __name__ == '__main__':
    # pay_off_str = '('
    # pay_off_str += ')'
    # a = game.payoff_matrix[0, 1, 1]
    # a_str = ', '.join(str(x) for x in a)
    # print(compute_marginal_pmf(np.array([0.14, 0.0, 0.86, 0.0])))
    # exit()
    
    payoff_matrix = np.zeros(shape=(2, 2, 2, 2, 4), dtype=int)
    
    # payoff_matrix[0,0,0,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,0,0,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[0,1,0,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,1,0,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[0,0,1,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,0,1,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[0,1,1,0] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,1,1,0] = np.array([ 0, 0, 0, 0])
    
    # payoff_matrix[0,0,0,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,0,0,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[0,1,0,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,1,0,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[0,0,1,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,0,1,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[0,1,1,1] = np.array([ 0, 0, 0, 0])
    # payoff_matrix[1,1,1,1] = np.array([ 0, 0, 0, 0])

    special_game = GeneralSumGame(4)
    for t in [3,7]:
        print('Game ', t)
        game = GeneralSumGame(4)
        strategy = algo_run_v4(game, save_id='4_player_game_same' + str(t))
    # print('Result:\n', strategy)
    # expected_payoff = NashEquilibriumJudger.get_payoff(strategy, game.payoff_matrix)
    # print(expected_payoff)
    # is_nash = NashEquilibriumJudger.run(strategy, game.payoff_matrix, n_steps=10000, epsilon=0.01)
    # print(is_nash)