import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# logdir = './log/test'
# nash_val = [5.2, -5.2]
# algos = ['iql', 'dynamic-nashq', 'nwqmix', 'ia2c']
# for i in range(1, 4):
#     test_case_n = str(i).zfill(3)
#     for algo in algos:
#         test_data_name = logdir + test_case_n + '-' + algo
#         print(test_data_name)

# 假設我們有多個實驗的數據
data = {
    "step": np.concatenate([np.arange(0, 2000000, 256) for _ in range(10)]),
    "reward": np.random.uniform(40, 100, 10 * (2000000 // 256)),  # 模擬 reward 數據
    "run": np.repeat(np.arange(10), 2000000 // 256)  # 代表不同的試驗組別
}
df = pd.DataFrame(data)

# 繪製曲線（帶置信區間）
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="step", y="reward", ci="sd")  # ci="sd" 表示標準差範圍
plt.xlabel("Training Steps")
plt.ylabel("Test Win Rate %")
plt.title("Training Progress")
plt.grid(True)

plt.show()