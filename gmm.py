import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
from water_filling import water_filling
import random

def generate_gmm_pdf(K, pi_k, mu_k, sigma2_k, x_values):
    """
    根據 GMM 的參數直接生成概率密度函數 (PDF)，而不依賴樣本數量。
    
    參數:
    K : int
        高斯分布的數量。
    pi_k : list of float
        各高斯分布的權重 (權重總和必須等於 1)。
    mu_k : list of float
        各高斯分布的均值。
    sigma2_k : list of float
        各高斯分布的方差。
    x_values : numpy array
        計算機率密度函數的 x 值範圍。

    返回:
    gmm_pdf : numpy array
        GMM 的機率密度值。
    """
    
    # 確保權重總和為1
    assert np.isclose(sum(pi_k), 1), "權重總和必須等於 1"
    assert len(pi_k) == K, "權重數量必須等於 K"
    assert len(mu_k) == K, "均值數量必須等於 K"
    assert len(sigma2_k) == K, "方差數量必須等於 K"

    # 初始化 GMM 的 PDF
    gmm_pdf = np.zeros_like(x_values)

    # 計算 GMM 的 PDF 為各分布的加權和
    for k in range(K):
        # 使用 scipy.stats.norm 計算每個分布的概率密度
        pdf_k = norm.pdf(x_values, mu_k[k], np.sqrt(sigma2_k[k]))
        gmm_pdf += pi_k[k] * pdf_k
    
    return gmm_pdf

def generate_gmm_pdf_for_users(K, pi_k, mu_k, sigma2_k, n_users):
    """
    根據 GMM 的參數生成對應於使用者數量的 GMM 機率密度函數 (PDF)。
    
    參數:
    K : int
        高斯分布的數量。
    pi_k : list of float
        各高斯分布的權重 (權重總和必須等於 1)。
    mu_k : list of float
        各高斯分布的均值。
    sigma2_k : list of float
        各高斯分布的方差。
    n_users : int
        使用者的總數。

    返回:
    gmm_pdf : numpy array
        GMM 機率密度值，長度為 n_users。
    """
    
    # 計算動態範圍
    min_x = max(0, min(mu_k) - 3 * np.sqrt(max(sigma2_k)))
    max_x = max(mu_k) + 3 * np.sqrt(max(sigma2_k))
    
    # 計算 x_values
    x_values = np.linspace(min_x, max_x, n_users)  # 產生 1024 個點的範圍
    gmm_pdf_full = generate_gmm_pdf(K, pi_k, mu_k, sigma2_k, x_values)

    # 根據用戶索引提取 PDF
    user_indices = np.linspace(0, len(gmm_pdf_full) - 1, n_users).astype(int)
    gmm_pdf_users = gmm_pdf_full[user_indices]
    
    return gmm_pdf_users

def resource_allocation(R, n_users, gmm_pdf):
    """
    根據 GMM 分布和資源總量 R, 為 n_users 分配資源。
    
    參數:
    R : int
        總資源數量，例如 1024。
    n_users : int
        使用者的總數。
    gmm_pdf : numpy array
        GMM 機率密度分布的結果。

    返回:
    resources : numpy array
        資源分配結果的數列，長度為 n_users, 保證 sum(resources) = R。
    """
    
    # 1. 根據 GMM 分布計算每個使用者的權重
    user_weights = gmm_pdf / np.sum(gmm_pdf)  # 正規化權重

    # 2. 根據權重比例分配資源 (先分配浮點數)
    raw_allocation = user_weights * R

    # 3. 四捨五入分配的資源並確保總和等於 R
    rounded_allocation = np.floor(raw_allocation).astype(int)
    remainder = R - np.sum(rounded_allocation)

    # 4. 分配剩餘的資源 (將剩下的資源分配給最高權重的使用者)
    if remainder > 0:
        indices_to_adjust = np.argsort(raw_allocation - rounded_allocation)[-remainder:]
        rounded_allocation[indices_to_adjust] += 1

    return rounded_allocation

def plot_allocation_vs_gmm(allocated_resources, gmm_pdf, R, n_users):
    """
    繪製實際資源分配和 GMM 分布的比較圖。

    參數:
    allocated_resources : numpy array
        資源分配結果的數列，長度為 n_users。
    gmm_pdf : numpy array
        GMM 機率密度分布的結果。
    R : int
        資源總量，例如 1024。
    n_users : int
        使用者的總數。
    """
    # 設定 x 軸的範圍
    x_values = np.arange(n_users)

    # 繪製柱狀圖表示分配的資源
    plt.bar(x_values, allocated_resources, color='lightblue', label='Allocated Resources', alpha=0.7, width=0.4)

    # 繪製折線圖表示 GMM 的 PDF
    # 為了匹配 x 軸範圍，對 GMM PDF 進行縮放
    gmm_pdf_scaled = gmm_pdf *np.max(allocated_resources)/np.max(gmm_pdf) # 將 GMM PDF 按 R 進行縮放
    
    plt.plot(x_values, gmm_pdf_scaled, color='red', label='GMM PDF', marker='o')#, linewidth=0.01)

    # 在柱狀圖的頂部顯示數值
    # for i, value in enumerate(allocated_resources):
    #     plt.text(i, value + (int)(np.max(gmm_pdf)*R*0.01), str(value), ha='center', va='bottom')

    # 設定圖形標題和標籤
    plt.title('Comparison of Allocated Resources and GMM Distribution')
    plt.xlabel('User Index')
    plt.ylabel('Resources Allocated (0 to R)')
    #plt.ylim(0, np.max(gmm_pdf)*R)
    plt.xlim(-2, n_users + 2)
    
    # 添加圖例
    plt.legend()
    
    # 顯示網格
    plt.grid(axis='y')
    
    # 顯示圖形
    plt.show()
if __name__ == "__main__":
    # 測試例子
    K = 4  # 三個高斯分布
    pi_k = [0.25, 0.25, 0.25, 0.25]  # 權重
    mu_k = [0, 0.4, 0.1, 0.7]  # 均值
    sigma2_k = [1, 2, 1.5, 2]  # 方差
    R = 1024*1024  # 資源總量
    n_users = 1024  # 使用者總數
    mu_k = [mu_i * n_users for mu_i in mu_k]
    
    t_start = time.time()
    # 根據使用者數量生成 GMM PDF
    gmm_pdf = generate_gmm_pdf_for_users(K, pi_k, mu_k, sigma2_k, n_users)

    # 資源分配
    allocated_resources = resource_allocation(R, n_users, gmm_pdf)

    print("資源分配結果:", allocated_resources)
    print("總資源分配:", np.sum(allocated_resources), R)  # 應該等於 R
    print("GMM運行時間:", time.time() - t_start)

    z = allocated_resources
    r = np.array(random.choices(list(range(0, (R+1)//2)), k=n_users))
    t_start = time.time()
    try:
        ans = water_filling(z, r, None, R)
        print("資源分配結果:", ans)
    except:
        print("--out of memory--")
    finally:
        print("Water-Filling運行時間:", time.time() - t_start)

    plot_allocation_vs_gmm(allocated_resources, gmm_pdf, R, n_users)
