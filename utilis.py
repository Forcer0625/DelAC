import numpy as np
from collections import deque
from copy import deepcopy

class FrameStack():
    def __init__(self, n_stacks:int=4):
        self.n_stacks = n_stacks
        self.frame_buffer = deque(maxlen=n_stacks)

    def get(self):
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        return stacked_frames

    def push(self, image:np.ndarray):
        self.frame_buffer.append(image)
        while len(self.frame_buffer) < self.n_stacks:
            self.frame_buffer.append(image)

    def render(self):
        pass

    def clear(self):
        self.frame_buffer.clear()
    
    def next_frame(self, image:np.ndarray):
        '''Return stacked frames the next frame'''
        temp = deepcopy(self.frame_buffer)
        temp.append(image)
        return np.stack(temp, axis=0)

class MultiFrameStack():
    def __init__(self, agents, n_stacks:int=4):
        self.frame_buffers = {}
        self.agents = agents
        for agent in self.agents:
            self.frame_buffers[agent] = FrameStack(n_stacks)

    def get(self):
        ''' ndarray: [agent, observation]'''
        observations = []
        for agent in self.agents:
            observations.append(self.frame_buffers[agent].get().reshape(-1))
        return np.array(observations)

    def push(self, state):
        if type(state) is dict:
            for agent in self.agents:
                self.frame_buffers[agent].push(state[agent])
        else:
            i = 0
            for agent in self.agents:
                self.frame_buffers[agent].push(state[i])
                i+=1

    def top(self):
        states = []
        for agent in self.agents:
            states.append(self.frame_buffers[agent].frame_buffer[-1])
        return np.array(states)

    def clear(self):
        for agent in self.agents:
            self.frame_buffers[agent].clear()
    
    def next_frame(self, single_frame):
        state = []
        observations = []
        for agent in self.agents:
            observations.append(self.frame_buffers[agent].next_frame(single_frame[agent]).reshape(-1))
            state.append(single_frame[agent])
        return np.array(state), np.array(observations)
        
    
class ReplayBuffer():
    def __init__(self, buffer_size:int, observation_shape, n_agents):
        self.max_buffer_size = buffer_size
        self.index = 0
        self.buffer_size = 0    #當前緩衝區中有效樣本的數量
        self.observations = np.zeros((self.max_buffer_size, n_agents, *observation_shape))  #儲存每個時間步的觀察值（狀態）。
        self.observations_= np.zeros((self.max_buffer_size, n_agents, *observation_shape))  #儲存執行動作後的下一觀察值。
        self.actions= np.zeros((self.max_buffer_size, n_agents), dtype=int) #儲存每個代理的動作。
        self.rewards= np.zeros((self.max_buffer_size, n_agents))
        self.dones  = np.zeros(self.max_buffer_size, dtype=bool)    #儲存每個時間步是否達到終止狀態（布林值）。

    def store(self, observation, action, reward, done, observation_):   
        store_index = self.index
        self.observations [self.index] = observation
        self.observations_[self.index] = observation_
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones  [self.index] = done

        self.index = (self.index+1)%self.max_buffer_size        #更新儲存位置，若超過緩衝區大小，則回到頭部（環狀緩衝區）。
        self.buffer_size = min(self.buffer_size+1, self.max_buffer_size)
        return store_index

    def sample(self, batch_size):
        batch = np.random.choice(self.buffer_size, batch_size)

        observations = self.observations [batch]
        observations_= self.observations_[batch] 
        actions= self.actions[batch]
        rewards= self.rewards[batch]
        dones  = self.dones  [batch]

        return observations, actions, rewards, dones, observations_
    
    def last_episode(self, episode_idx):
        observations = self.observations [episode_idx]
        observations_= self.observations_[episode_idx] 
        actions= self.actions[episode_idx]
        rewards= self.rewards[episode_idx]
        dones  = self.dones  [episode_idx]

        return observations, actions, rewards, dones, observations_
    
    def __len__(self):
        return self.buffer_size
    
class RunningMeanStd():
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x:np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class Normalization():
    def __init__(self, shape, n_env=1, use_clip=False):
        self.running_ms = RunningMeanStd(shape=shape)
        self.n_env = n_env
        self.use_clip = use_clip

    def __call__(self, x:np.ndarray, update=True):
        # Whether to update the mean and std, during the evaluating, update=False
        if update:
            self.running_ms.update(x)
        if self.use_clip:
            x = np.clip((x - self.running_ms.mean) / np.sqrt(self.running_ms.var + 1e-8), -10.0, 10.0)
        else:
            x = (x - self.running_ms.mean) / np.sqrt(self.running_ms.var + 1e-8)

        return x

class RewardScaling(Normalization):
    def __call__(self, x:np.ndarray):      
        self.running_ms.update(x)
        if self.use_clip:
            x = np.clip(x / np.sqrt(self.running_ms.var + 1e-8), -10.0, 10.0)
        else:
            x = x / np.sqrt(self.running_ms.var + 1e-8)
        return x