from qmix import *
from iql import IQL, NashQ
from ia2c import IA2C
from ca2c import CA2C
import torch
from envs import *
from multi_env import make_env

total_steps = int(1e5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'total_steps':total_steps,
    'eps_start':0.99,
    'eps_end':0.05,
    'eps_dec':total_steps*0.35, # more leads to slower decay
    'gamma':0.99,
    'lr': 3e-4,
    'tau':100, # more is harder
    'batch_size':1024,
    'memory_size':4096,
    'device':device,
    'nash-dynamic':True,
    'use-parameter-sharing':True,
    'feasibility':True,
    'logdir':'test000-dynamic-nashq',
}

ac_config = {
    'lam':0.95,
    'ent_coef':0.05,
    'n_env':4,
    'batch_size':256,
    "grad_norm":0.5,
}
ac_config['print_every'] = total_steps//config['batch_size']//ac_config['n_env']//10 + 1

if __name__ == '__main__':
    env = GMP(w=0.5)#TwoTeamZeroSumSymmetricStochasticEnv(n_states=1, n_agents=4, n_actions=2)
    config['n_states'] = env.n_states
    config['n_agents'] = env.n_agents
    config['n_actions'] = env.n_actions
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

    config.update(ac_config)
    config['logdir'] = config['logdir'].replace('nash', 'ia2c')
    envs = MultiEnv([make_env(i, deepcopy(env), config) for i in range(config['n_env'])])
    runner = OnPolicyRunner(envs, config)
    ia2c = IA2C(runner, config)
    ia2c.learn(total_steps)
    with torch.no_grad():
        obs = torch.as_tensor([0]).float().to(config['device'])
        team_1_a_porbs = ia2c.actor_critic[ 0].actor.model(obs)
        team_2_a_porbs = ia2c.actor_critic[-1].actor.model(obs)
    ia2c.config['team 1 strategy'] = [str(i) for i in list(team_1_a_porbs.cpu().numpy())]
    ia2c.config['team 2 strategy'] = [str(i) for i in list(team_2_a_porbs.cpu().numpy())]
    ia2c.save_config()

    config['logdir'] = config['logdir'].replace('ia2c', 'ca2c')
    runner = CentralisedOnPolicyRunner(envs, config)
    ca2c = CA2C(runner, config)
    ca2c.learn(total_steps)
    with torch.no_grad():
        obs = torch.as_tensor([0]).float().to(config['device'])
        team_1_a_porbs = ca2c.actors[ 0].model(obs)
        team_2_a_porbs = ca2c.actors[-1].model(obs)
    ca2c.config['team 1 strategy'] = [str(i) for i in list(team_1_a_porbs.cpu().numpy())]
    ca2c.config['team 2 strategy'] = [str(i) for i in list(team_2_a_porbs.cpu().numpy())]
    print(team_1_a_porbs)
    print(team_2_a_porbs)
    ca2c.save_config()