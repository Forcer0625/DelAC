from qmix import *
from iql import IQL, NashQ
import torch
from envs import *

total_steps = int(1e5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'total_steps':total_steps,
    'eps_start':0.99,
    'eps_end':0.05,
    'eps_dec':total_steps*0.33, # more leads to slower decay
    'gamma':0.99,
    'lr': 3e-4,
    'tau':100, # more is harder
    'batch_size':1024,
    'memory_size':4096,
    'device':device,
    'nash-dynamic':True,
    'use-parameter-sharing':True,
    'feasibility':True,
    'logdir':'test001-dynamic-nashq',
}

if __name__ == '__main__':
    env = TwoTeamZeroSumSymmetricStochasticEnv(n_states=1, n_agents=4, n_actions=2)
    config['n_states'] = env.n_states
    config['n_agents'] = env.n_agents
    config['n_ations'] = env.n_actions
    print(config)

    infos = []
    nashq = NashQ(env, config)
    nashq.learn(total_steps)
    infos.append(nashq.extract_q())

    config['logdir'] = config['logdir'].replace('dynamic-nashq', 'iql')
    iql = IQL(env, config)
    iql.learn(total_steps)
    infos.append(iql.extract_q())

    config['logdir'] = config['logdir'].replace('iql', 'nwqmix')
    nwqix = NWQMix(env, config)
    nwqix.learn(total_steps)
    infos.append(nwqix.extract_q())

    config['logdir'] = config['logdir'].replace('nwqmix', 'nash')
    env.save(infos, config)