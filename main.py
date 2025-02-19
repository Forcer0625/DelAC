from qmix import *
from iql import IQL, NashQ
import torch
from envs import *

total_steps = int(7e5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'eps_start':0.99,
    'eps_end':0.05,
    'eps_dec':total_steps*0.33, # more leads to slower decay
    'gamma':0.99,
    'lr': 1e-3,
    'tau':0.005, # more is harder
    'batch_size':256,
    'memory_size':40000,
    'device':device,
    'nash-dynamic':True,
    'logdir':'dynamic-nashq-test236',
}

if __name__ == '__main__':
    print(device)
    env = StochasticGame(n_states=5, n_agents=4, n_actions=2)
    nashq = NashQ(env, config)
    nashq.learn(total_steps)

    config['logdir'] = config['logdir'].replace('dynamic-nashq', 'iql')
    iql = IQL(env, config)
    iql.learn(total_steps)
    
    # iql.save_model()