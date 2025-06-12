from qmix import *
from iql import IQL, NashQ
from ia2c import IA2C
from ca2c import CA2C, CFAC
from ffq import FFQ
from ippo import IPPO
from mappo import MAPPO
from ceq import CEQ
import torch
from envs import *
from multi_env import make_env

runs_data_path = './runs/'
run_case = '250329-YF_GeneralSum'
run_env = run_case[7:]
nash_file_postfix = '-nash/data.csv'
    
def train(env_name, training_num, w=0.5, path=None):
    total_steps = int(5e4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'total_steps':total_steps,
        'eps_start':0.99,
        'eps_end':0.05,
        'eps_dec':total_steps*0.35, # more leads to slower decay
        'gamma':0.99,
        'lr': 3e-4,
        'tau':100, # more is harder
        'batch_size':128,
        'memory_size':4096,
        'device':device,
        'nash-dynamic':True,
        'use-parameter-sharing':True,
        'feasibility':True,
        'logdir':'test000-dynamic-nashq',
    }

    ac_config = {
        'lam':0.95,
        'ent_coef':0.0,
        'n_env':4,
        'batch_size':256,
        'grad_norm':0.5,
    }
    ppo_config = {
        'clip_param':0.2,
        'epochs':4,
        'sample_mb_size':ac_config['batch_size']*ac_config['n_env'],
    }
    ac_config['print_every'] = 5120 + 1
    config['logdir'] = 'noise(0,1)-'+env_name + str(training_num).zfill(3) + '-nash'
    

    class NoiseSampler():
        def __init__(self, loc, scale):
            self.loc = loc
            self.scale = scale
            
        def sample(self, size=None):
            return np.random.normal(self.loc, self.scale, size=size)

    if 'GMP' in env_name:
        env = GMP(w=w)
        config['gmp_w'] = str(w)
    elif 'ZeroSum' in env_name:
        env = TwoTeamZeroSumSymmetricStochasticEnv(n_states=1, n_agents=4, n_actions=2, noise=NoiseSampler(0.0, 1.0))
    elif 'GeneralSum' in env_name:
        env = TwoTeamSymmetricStochasticEnv(n_states=1, n_agents=4, n_actions=2, noise=NoiseSampler(0.0, 1.0))

    if path is not None and 'GMP' not in env_name:
        csv_file_name = runs_data_path+run_case+'/'+run_env+str(training_num).zfill(3)+nash_file_postfix
        env = env.from_csv(csv_file_name)

    config['n_states'] = env.n_states
    config['n_agents'] = env.n_agents
    config['n_actions'] = env.n_actions
    print(config)

    # infos = []
    # valid = env.save(infos, config)
    # if not valid:
    #     return False

    # config['logdir'] = config['logdir'].replace('nash', 'ceq')
    # ceq = CEQ(env, config)
    # ceq.learn(total_steps)
        
    config['logdir'] = config['logdir'].replace('nash', 'ffq(foe)')
    ffq = FFQ(env, config, 'foe')
    ffq.learn(total_steps)
    if 'GeneralSum' in env_name:
        config['logdir'] = config['logdir'].replace('foe', 'friend')
        ffq = FFQ(env, config, 'friend')
        ffq.learn(total_steps)
        config['logdir'] = config['logdir'].replace('friend', 'foe')
    
    config['logdir'] = config['logdir'].replace('ffq(foe)', 'dynamic-nashq')
    nashq = NashQ(env, config)
    nashq.learn(total_steps)

    config['logdir'] = config['logdir'].replace('dynamic-nashq', 'iql')
    iql = IQL(env, config)
    iql.learn(total_steps)

    config['logdir'] = config['logdir'].replace('iql', 'nwqmix')
    nwqix = NWQMix(env, config)
    nwqix.learn(total_steps)

    config.update(ac_config)
    config['logdir'] = config['logdir'].replace('nwqmix', 'ia2c')
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
    envs.close()

    config['logdir'] = config['logdir'].replace('ia2c', 'ca2c')
    envs = MultiEnv([make_env(i, deepcopy(env), config) for i in range(config['n_env'])])
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
    ca2c.save_game(TwoTeamSymmetricGame)
    ca2c.save_config()
    envs.close()

    config['logdir'] = config['logdir'].replace('ca2c', 'cfac')
    envs = MultiEnv([make_env(i, deepcopy(env), config) for i in range(config['n_env'])])
    runner = CentralisedOnPolicyRunner(envs, config)
    cfac = CFAC(runner, config)
    cfac.learn(total_steps)
    with torch.no_grad():
        obs = torch.as_tensor([0]).float().to(config['device'])
        team_1_a_porbs = cfac.actors[ 0].model(obs)
        team_2_a_porbs = cfac.actors[-1].model(obs)
    cfac.config['team 1 strategy'] = [str(i) for i in list(team_1_a_porbs.cpu().numpy())]
    cfac.config['team 2 strategy'] = [str(i) for i in list(team_2_a_porbs.cpu().numpy())]
    print(team_1_a_porbs)
    print(team_2_a_porbs)
    cfac.save_game(TwoTeamSymmetricGame)
    cfac.save_config()
    envs.close()

    config.update(ppo_config)
    config['logdir'] = config['logdir'].replace('nash', 'ippo')
    envs = MultiEnv([make_env(i, deepcopy(env), config) for i in range(config['n_env'])])
    runner = PPORunner(envs, config)
    ippo = IPPO(runner, config)
    ippo.learn(total_steps)
    with torch.no_grad():
        obs = torch.as_tensor([0]).float().to(config['device'])
        team_1_a_porbs = ippo.actor_critic[ 0].actor.model(obs)
        team_2_a_porbs = ippo.actor_critic[-1].actor.model(obs)
    ippo.config['team 1 strategy'] = [str(i) for i in list(team_1_a_porbs.cpu().numpy())]
    ippo.config['team 2 strategy'] = [str(i) for i in list(team_2_a_porbs.cpu().numpy())]
    ippo.save_config()
    envs.close()
    
    config['logdir'] = config['logdir'].replace('ippo', 'mappo')
    envs = MultiEnv([make_env(i, deepcopy(env), config) for i in range(config['n_env'])])
    runner = CentralisedPPORunner(envs, config)
    mappo = MAPPO(runner, config)
    mappo.learn(total_steps)
    with torch.no_grad():
        obs = torch.as_tensor([0]).float().to(config['device'])
        team_1_a_porbs = mappo.actors[ 0].model(obs)
        team_2_a_porbs = mappo.actors[-1].model(obs)
    mappo.config['team 1 strategy'] = [str(i) for i in list(team_1_a_porbs.cpu().numpy())]
    mappo.config['team 2 strategy'] = [str(i) for i in list(team_2_a_porbs.cpu().numpy())]
    mappo.save_config()
    envs.close()

    return True

if __name__ == '__main__':
    env_name = 'YF_GeneralSum'
    run_case = '250329-YF_GeneralSum'
    run_env = run_case[7:]
    for i in range(30):
        print(str(i+1).zfill(3)+':training...'+env_name)
        valid = train(env_name, i+1, 0.5, True)
        if not valid:
            i = i - 1

    env_name = 'ZeroSum'
    run_case = '250326-ZeroSum'
    run_env = run_case[7:]
    for i in range(30):
        print(str(i+1).zfill(3)+':training...'+env_name)
        valid = train(env_name, i+1, 0.5, True)
        if not valid:
            i = i - 1