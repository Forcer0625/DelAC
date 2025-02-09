import numpy as np

def water_filling(z:np.array, r:np.array, m:np.array, L) -> np.array:
    if L >= r.sum():
        return r
    #1.~2.
    non_zero_users = np.nonzero(r)[0]
    if len(non_zero_users) == 0:
        return z
    N_user_id = np.argsort(z[non_zero_users], kind="stable")
    k = len(z[non_zero_users]) - np.argmin(np.flip(z[non_zero_users][N_user_id]))
    #3.
    requset_user_id = N_user_id[np.argsort(z[non_zero_users][N_user_id][:k], kind="stable")]
    requset_user_id = N_user_id[np.argsort(r[non_zero_users][requset_user_id], kind="stable")]
    #4.
    try:
        threshold = z[non_zero_users][N_user_id[k]] - z[non_zero_users][N_user_id[k-1]]
        m_ = min(r[non_zero_users][requset_user_id[0]], threshold)
    except:
        threshold = 0
        m_ = r[non_zero_users][requset_user_id[0]]
    #5.~9.
    if L <= m_*k:
        x = np.zeros(len(z), dtype=np.int64)
        users = non_zero_users[N_user_id[:k]]
        x[users] = L//k

        L -= len(users)*(L//k)
        
        id = 0
        while L>0:
            L -= 1
            x[users[id]] += 1
            id += 1
        return x
    #10.~11.
    users = non_zero_users[N_user_id[:k]]
    z_ = z.copy()
    z_[users] += m_
    r_ = r.copy()
    r_[users] -= m_
    #12.
    L_ = L - m_*k
    #13.
    y = water_filling(z_, r_, m, L_)
    #14.~19.
    x = y.copy()
    if r[non_zero_users][requset_user_id[0]] < threshold:
        j = 0
        while j < len(requset_user_id) and r[non_zero_users][requset_user_id[0]] >= r[non_zero_users][requset_user_id[j]]:
            j += 1
        users = non_zero_users[requset_user_id]
        x[users[ :j]] = m_
        x[users[j:k]] += m_
    else:
        users = non_zero_users[N_user_id[:k]]
        x[users] += m_

    return x

if __name__ == "__main__":
    z = np.array([1,0,3,0,1])
    r = np.array([3,1,0,1,6])
    L = 5
    ans = water_filling(z, r, None, L)
    z_ = z+ans
    print(ans)
    print("next z:", z_)
    print(np.var(z))