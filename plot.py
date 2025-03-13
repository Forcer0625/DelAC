import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as p
import torch

logdir = './log/test'
# test013-020 50k steps
nash_val = [
    [5.963, -5.963],
    [-1.0, 1.0],
    [2.0, -2.0],
    [2.377, -2.377],
    [-0.437, 0.437],
    [2.557, -2.557],
    [-0.201, 0.201],
    [-3.019, 3.019],
]
algos = ['iql', 'dynamic-nashq', 'nwqmix']#, 'ia2c']
for algo in algos:
    for n_test in range(13, 20):
        test_case_n = str(n_test).zfill(3)
        test_data_name = logdir + test_case_n + '-' + algo
        training_data = torch.load(test_data_name)
print(len(training_data))
exit()
# 模擬數據
np.random.seed(42)
steps = np.arange(0, 2000000, 256)
#runs = 10  # 每個演算法的試驗數量

data = []
for algo in algos:
    for run in range(13, 20):
        se = np.random.uniform(40, 100, len(steps))  # 模擬 reward 數據
        for step, reward in zip(steps, se):
            data.append([step, reward, algo, run])

df = pd.DataFrame(data, columns=["step", "reward", "algorithm", "run"])

# 繪製多條曲線
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="step", y="reward", hue="algorithm", errorbar="sd")  # 每個演算法不同顏色
plt.xlabel("Training Steps")
plt.ylabel("MSE of Nash Equilibrium Expected Payoff")
plt.legend(title="Algorithm")
#plt.title("Training Progress")
plt.grid(True)

plt.show()