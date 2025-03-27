import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from collections import deque

def smooth_tensorboard(data, alpha=0.6):
    """ 
    TensorBoard-style exponential moving average smoothing.
    
    Args:
    - data (pd.Series): 要平滑的數據序列
    - alpha (float): 平滑係數 (0.0 ~ 1.0)
    
    Returns:
    - pd.Series: 平滑後的數據
    """
    smoothed = []
    last = data.iloc[0]  # 初始化為第一個數據點
    for value in data:
        last = last * (1 - alpha) + alpha * value
        smoothed.append(last)
    return pd.Series(smoothed, index=data.index)

logdir = './log/'
# test013-032 50k steps
nash_val = [
    [5.963, -5.963],
    [-1.0, 1.0],
    [2.0, -2.0],
    [2.377, -2.377],
    [-0.437, 0.437],
    [2.557, -2.557],
    [-0.201, 0.201],
    [-3.019, 3.019],
    [-4.222, 4.222],
    [-4.198, 4.198],
    [-6.045, 6.045],
    [0.318, -0.318],
    [1.467, -1.467],
    [-2.426, 2.426],
    [0.318, -0.318],
    [1.793, -1.793],
    [-0.74, 0.74],
    [-0.524, 0.524],
    [2.484, -2.484],
    [4.085, -4.085],
    [-1.641, 1.641],
]
# test033-055 50k steps
nash_val = [
    [2.0, 5.719],
    [7.66, 2.475],
    [5.632, 6.252],
    [5.149,5.878],
    [5.565,5.944],
    [7.274,3.181],
    [3.152,4.0],
    [3.725,5.601],
    [2.335,5.555],
    [3.758,6.207],
    [4.956,2.658],
    [5.199,2.778],
    [3.631,5.346],
    [4.846,5.523],
    [6.225,8.157],
    [5.8,5.229],
    [4.054,6.21],
    [3.646,3.836],
    [7.222,7.838],
    [4.865,2.842],
    [4.523,4.388],
    [5.171,5.976],
    [5.947,4.62],
]
nash_val = {}
# GMP(w=0.5)
nash_val['GMP(w=0.5)'] = [
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
]
# ZeroSum
nash_val['ZeroSum'] = [
    [-0.366, 0.366],
    [-2.998, 2.998],
    [-0.983, 0.983],
    [-0.596, 0.596],
    [-0.003, 0.003],
    [0.874, -0.874],
    [0.677, -0.677],
    [3.8, -3.8],
    [3.5, -3.5],
    [1.952, -1.952],
    [-1.684, 1.684],
    [-2.189, 2.189],
    [0.002, -0.002],
    [5.298, -5.298],
    [2.25, -2.25],
    [0.435, -0.435],
    [0.57, -0.57],
    [-3.672, 3.672],
    [-0.997, 0.997],
    [-0.336, 0.336],
    [0.547, -0.547],
    [0.658, -0.658],
    [0.811, -0.811],
    [3.75, -3.75],
    [-2.593, 2.593],
    [-1.142, 1.142],
    [3.994, -3.994],
    [0.775, -0.775],
    [0.7, -0.7],
    [-3.0, 3.0],
]
# GeneralSum
nash_val['GeneralSum']  = [
    [5.423,3.034],
    [7.8,6.4],
    [6.786,8.694],
    [2.83,3.123],
    [7.062,2.25],
    [3.125,5.52],
    [6.163,3.871],
    [8.027,2.319],
    [3.0,7.0],
    [10.0,4.0],#10
    [3.476,5.531],
    [5.023,6.475],
    [4.538,4.602],
    [5.215,5.97],
    [3.964,3.118],
    [7.037,6.142],
    [4.354,5.088],
    [6.562,3.25],
    [7.667,1.333],
    [2.398,3.458],#20
    [10.0,3.0],
    [4.788,6.18],
    [2.0,5.779],
    [5.847,4.281],
    [5.339,4.103],
    [7.049,4.41],
    [4.375,5.25],
    [5.428,7.455],
    [6.64,5.809],
    [3.24,3.199], #30
]

algos = ['cfac', 'ia2c', 'ca2c', 'iql', 'dynamic-nashq', 'nwqmix']
env_name = 'ZeroSum'
data = []
#reward_buf = deque(maxlen=100)
start_idx = 1
for algo in algos:
    print(algo+'...')
    for n_test in range(start_idx, 31):
        test_case_n = str(n_test).zfill(3)
        test_data_name = logdir + env_name + test_case_n + '-' + algo
        training_data = torch.load(test_data_name)
        n_steps = len(training_data)
        for info in training_data:
            mse = (info['Team1-Ep.Reward'] - nash_val[env_name][n_test-start_idx][0])**2
            mse += (info['Team2-Ep.Reward'] - nash_val[env_name][n_test-start_idx][1])**2
            mse = mse / 2.0
            step = info['Step']
            # reward_buf.append((training_data[step]['Team1-Ep.Reward'] - nash_val[n_test-start_idx][0])**2)
            # reward_buf.append((training_data[step]['Team2-Ep.Reward'] - nash_val[n_test-start_idx][1])**2)
            # mse = np.mean(reward_buf, axis=0)
            data.append([step, mse, algo.upper() if algo!='dynamic-nashq' else 'NashQ' , n_test])

print('sampling...')
# 模擬數據
# np.random.seed(42)
# steps = np.arange(0, 2000000, 256)
# #runs = 10  # 每個演算法的試驗數量

# data = []
# for algo in algos:
#     for run in range(13, 20):
#         se = np.random.uniform(40, 100, len(steps))  # 模擬 reward 數據
#         for step, reward in zip(steps, se):
#             data.append([step, reward, algo, run])

df = pd.DataFrame(data, columns=["step", "reward", "algorithm", "run"])
# 降采樣 (每 100 個 step 取 1 個)
df_sampled = df[df["step"] % 256 == 0].copy() 

print('smooth...')
# 平滑處理
# df_sampled.loc[:, "reward_smooth"] = df_sampled.groupby("algorithm")["reward"].transform(
#     lambda x: smooth_tensorboard(x, alpha=0.9)
# )

print('plot...')
# 繪製多條曲線
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_sampled, x="step", y="reward", hue="algorithm")#, errorbar="sd")  # 每個演算法不同顏色
plt.xlabel("Training Steps")
plt.ylabel("MSE of Nash Equilibrium Expected Payoff")
plt.legend(title="Algorithm")
plt.grid(True)

plt.show()