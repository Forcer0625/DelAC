import numpy as np
from envs import *
from multiprocessing import Process, Pipe

#-----------------------
# Make an environment
#-----------------------
def make_env(rank, env, config:dict, rand_seed=None):
    def _thunk():
        if 'w' in config.keys():
            return env(config['w'])
        return env
        #return env(config['n_states'], config['n_agents'], config['n_actions'], seed=rand_seed)
    return _thunk

#-----------------------
# Worker
#-----------------------
def worker(remote, parent_remote, env_fn_wrapper):
	parent_remote.close()
	env = env_fn_wrapper.x()

	while True:
		cmd, data = remote.recv()

		if cmd == "step":
			obs, reward, done, truncation, info = env.step(data)
			if done or truncation:
				obs, _ = env.reset()

			remote.send((obs, reward, done, truncation, info))

		elif cmd == "reset":
			obs, info = env.reset()
			remote.send((obs, info))

		elif cmd == "close":
			remote.close()
			break

		else:
			raise NotImplementedError

#To serialize contents (otherwise multiprocessing tries to use pickle)
class CloudpickleWrapper():
	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)

#Multiple environment
class MultiEnv():
	#-----------------------
	# Constructor
	#-----------------------
	def __init__(self, env_fns):
		self.closed = False
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
		self.subprocs = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
						for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
		self.n_env = len(self.remotes)

		#Start subprocesses
		for p in self.subprocs:
			p.deamon = True
			p.start()

		for remote in self.work_remotes:
			remote.close()

	#-----------------------
	# Step
	#-----------------------
	def step(self, actions):
		for remote, action in zip(self.remotes, actions):
			remote.send(("step", action))

		results = [remote.recv() for remote in self.remotes]
		obs, rewards, dones, truncations, infos = zip(*results)

		return np.stack(obs), np.stack(rewards), np.stack(dones), np.stack(truncations), infos

	#-----------------------
	# Reset
	#-----------------------
	def reset(self):
		for remote in self.remotes:
			remote.send(("reset", None))
		
		results = [remote.recv() for remote in self.remotes]
		obs, infos = zip(*results)

		return np.stack(obs), infos
	#	return np.stack([remote.recv() for remote in self.remotes])

	#-----------------------
	# Close
	#-----------------------
	def close(self):
		if self.closed: return

		for remote in self.remotes:
			remote.send(("close", None))

		for p in self.subprocs:
			p.join()

		self.closed = True