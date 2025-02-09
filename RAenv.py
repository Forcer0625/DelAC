import numpy as np
import random
from gmm import generate_gmm_pdf_for_users, resource_allocation, plot_allocation_vs_gmm
from water_filling import water_filling

class SimpleResourceAllocation:
    def __init__(self, n_users=100, L_max=1000, max_steps=1e6):
        self.n_users = n_users
        self.L_max = L_max
        self.max_steps = (int)(max_steps)

    def reset(self, seed=None):
        random.seed(seed)
        self.z = np.zeros(self.n_users, dtype=int)
        self.rng = np.random.default_rng(seed)
        self.r = self.rng.integers(self.L_max, size=self.n_users, dtype=self.z.dtype)
        self.steps_count = 0
        
    def step(self, allocation):
        self.steps_count += 1
        if np.sum(allocation) > self.L_max:
            pass
            raise ValueError
        
        demand = allocation - self.r
        request_fullfilled = np.where(demand > 0)[0]

        self.z = self.z + allocation
        z_norm = (self.z - self.z.mean())/(self.z.std())
        #softmax_z = np.exp(z_norm)/sum(np.exp(z_norm))
        sample_var = np.var(self.z, dtype=np.float32)
        shaped_var = self.z.std()/self.n_users#sample_var/self.L_max/self.z.mean()*self.n_users
        reward -= shaped_var
        #print(np.var(softmax_z))
        #print(np.sum(self.z))

        self.r = self.rng.integers(self.L_max, size=self.n_users, dtype=self.z.dtype)

        obs = np.concatenate([self.z, self.r])

        termination = truncation = self.steps_count > self.max_steps
        reward = (float)(len(request_fullfilled))/self.n_users

        return obs, reward, termination, truncation, {'sample_varience': sample_var,\
                                                      'shaped_varience': shaped_var}


if __name__ == "__main__":
    env = SimpleResourceAllocation(n_users=20)
    env.reset()
    
    for _ in range(10):
        # 測試例子
        K = 3  # 三個高斯分布
        pi_k = [0.2, 0.5, 0.3]  # 權重
        mu_k = [0, 5, 10]  # 均值
        sigma2_k = [1, 2, 1.5]  # 方差
        
        # 根據使用者數量生成 GMM PDF
        gmm_pdf = generate_gmm_pdf_for_users(K, pi_k, mu_k, sigma2_k, env.n_users)

        # 資源分配
        allocated_resources = resource_allocation(env.L_max, env.n_users, gmm_pdf)

        obs, reward, _, _, info = env.step(allocated_resources)#water_filling(env.z, env.r, None, env.L_max))
        print(reward, info['shaped_varience'])#, np.sum(allocated_resources))
    plot_allocation_vs_gmm(allocated_resources, gmm_pdf, env.L_max, env.n_users)