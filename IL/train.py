#!/usr/bin/env python3

import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
import math

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from modules.cost_encoder import get_cost_encoder

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec[cfg.obs_type].shape
	cfg.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.train_env.observation_spec(),
								self.train_env.action_spec(), cfg.agent)

		if repr(self.agent) == 'drqv2':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_drq
		if repr(self.agent) == 'bc':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
			self.cfg.suite.num_seed_frames = 0

		if self.cfg.adaptive_discount:
			self.cfg.ads.horizon = self.env_horizon
			self.ads = hydra.utils.instantiate(self.cfg.ads)

		self.expert_replay_loader = make_expert_replay_loader(
			self.cfg.expert_dataset, self.cfg.expert_batch_size, self.cfg.num_demos, self.cfg.obs_type)
		self.expert_replay_iter = iter(self.expert_replay_loader)
			
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

		# expert_pixel is 224*224, expert_demo is 84*84
		with open(self.cfg.expert_dataset, 'rb') as f:
			data = pickle.load(f)
			if self.cfg.obs_type == 'pixels':
				self.expert_demo, _, self.expert_action, self.expert_reward, self.expert_pixel = data
				self.expert_pixel = self.expert_pixel[:self.cfg.num_demos]
			elif self.cfg.obs_type == 'features':
				_, self.expert_demo, self.expert_action, self.expert_reward, self.expert_pixel = data
		self.expert_demo = self.expert_demo[:self.cfg.num_demos]
		self.expert_action = self.expert_action[:self.cfg.num_demos]
		self.expert_reward = np.mean(self.expert_reward[:self.cfg.num_demos])

		if self.cfg.obs_type == 'pixels':
			for i in range(len(self.expert_pixel)):
				self.expert_pixel[i] = self.expert_pixel[i][::self.cfg.suite.action_repeat]
			cost_encoder = get_cost_encoder(self.cfg.cost_encoder, self.cfg.device)
			with torch.no_grad():
				demos = [cost_encoder(torch.tensor(demo).to(self.device)) for demo in self.expert_pixel]
			if repr(self.agent) == 'ot':
				self.agent.init_demos(demos, cost_encoder)
			if self.cfg.adaptive_discount:
				self.ads.init_demos(demos, 'pixels', cost_encoder)
		else:
			for i in len(self.expert_demo):
				self.expert_demo[i] = self.expert_demo[i][::self.cfg.suite.action_repeat]
			if repr(self.agent) == 'ot':
				self.agent.init_demos(self.expert_demo, None)
			if self.cfg.adaptive_discount:
				self.ads.init_demos(demos, 'features', None)
		
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		self.train_env, self.env_horizon = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.eval_env, self.env_horizon = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.env_horizon = math.ceil((self.env_horizon - 1) / self.cfg.suite.action_repeat) + 1

		# create replay buffer
		data_specs = [
			self.train_env.observation_spec()[self.cfg.obs_type],
			self.train_env.action_spec(),
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount')
		]

		self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer')

		self.replay_loader = make_replay_loader(
			self.work_dir / 'buffer', self.cfg.replay_buffer_size,
			self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
			self.cfg.save_experiences, self.cfg.nstep, self.cfg.suite.discount)

		self._replay_iter = None
		self.expert_replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat

	@property
	def replay_iter(self):
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		step, episode, total_reward = 0, 0, 0
		eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

		if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
			paths = []
		costs = []
		while eval_until_episode(episode):
			if self.cfg.suite.name == 'metaworld':
				path = []
			time_step = self.eval_env.reset()
			observations = []
			pixels = []
			self.video_recorder.init(self.eval_env, enabled=(episode == 0))
			episode_step = 0
			while not time_step.last():
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(time_step.observation[self.cfg.obs_type], self.global_step, eval_mode=True)
				observations.append(time_step.observation[self.cfg.obs_type])
				pixels.append(time_step.observation['pixels_large'])
				time_step = self.eval_env.step(action)
				if self.cfg.suite.name == 'metaworld':
					path.append(time_step.observation['goal_achieved'])
				self.video_recorder.record(self.eval_env)
				total_reward += time_step.reward
				step += 1
				episode_step += 1

			episode += 1
			self.video_recorder.save(f'{self.global_frame}.mp4')
			if self.cfg.suite.name == 'openaigym':
				paths.append(time_step.observation['goal_achieved'])
			elif self.cfg.suite.name == 'metaworld':
				paths.append(1 if np.sum(path)>3 else 0)

			observations = np.stack(observations, 0)
			pixels = np.stack(pixels, 0)
			if self.cfg.adaptive_discount:
				if self.cfg.obs_type == 'features':
					reward_obs = observations
				else:
					reward_obs = pixels
				costs += self.ads.compute_cost(reward_obs)
								
		if self.cfg.adaptive_discount:
			discount, ads_metrics = self.ads.update(costs)
			self.replay_storage.update_parameters({'_discount': discount})

		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
			if repr(self.agent) != 'drqv2':
				log('expert_reward', self.expert_reward)
			if self.cfg.suite.name == 'openaigym' or self.cfg.suite.name == 'metaworld':
				log("success_percentage", np.mean(paths))
			if self.cfg.adaptive_discount:
				for k, v in ads_metrics.items():
					log(k, v)
		
		if self.cfg.save_every_model:
			save_dir = self.work_dir / 'models'
			save_dir.mkdir(exist_ok=True)
			self.save_snapshot(save_dir / f'snapshot{self.global_frame}.pt')

	def train_il(self):
		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0
		time_steps = list()
		observations = list()
		pixels = list()
		actions = list()

		time_step = self.train_env.reset()
		time_steps.append(time_step)
		actions.append(time_step.action)
		
		if repr(self.agent) == 'ot':
			if self.agent.auto_rew_scale:
				self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

		self.train_video_recorder.init(time_step.observation['pixels'])
		metrics = None
		while train_until_step(self.global_step):
			if time_step.last():
				self._global_episode += 1
				if self._global_episode % 1 == 0:
					self.train_video_recorder.save(f'{self.global_frame}.mp4')
				# wait until all the metrics schema is populated
				observations = np.stack(observations, 0)
				pixels = np.stack(pixels, 0)
				if self.cfg.obs_type == 'features':
					reward_obs = observations
				else:
					reward_obs = pixels
				actions = np.stack(actions, 0)
				if repr(self.agent) == 'ot':
					new_rewards = self.agent.ot_rewarder(reward_obs, self.global_step)
					new_rewards_sum = np.sum(new_rewards)
				elif repr(self.agent) == 'gaifo':
					new_rewards = self.agent.gaifo_rewarder(observations, actions)
					new_rewards_sum = np.sum(new_rewards)
				
				if repr(self.agent) == 'ot':
					if self.agent.auto_rew_scale: 
						if self._global_episode == 1:
							self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(np.abs(new_rewards_sum))
							new_rewards = self.agent.ot_rewarder(reward_obs, self.global_step)
							new_rewards_sum = np.sum(new_rewards)

				for i, elt in enumerate(time_steps):
					elt = elt._replace(
						observation=time_steps[i].observation[self.cfg.obs_type])
					if repr(self.agent) == 'ot' or repr(self.agent) == 'gaifo':
						if i == 0:
							elt = elt._replace(reward=float('nan'))
						else:
							elt = elt._replace(reward=new_rewards[i - 1])
					self.replay_storage.add(elt)

				if metrics is not None:
					# log stats
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.suite.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
						log('fps', episode_frame / elapsed_time)
						log('total_time', total_time)
						log('episode_reward', episode_reward)
						log('episode_length', episode_frame)
						log('episode', self.global_episode)
						log('buffer_size', len(self.replay_storage))
						log('step', self.global_step)
						if repr(self.agent) == 'ot' or repr(self.agent) == 'gaifo':
								log('expert_reward', self.expert_reward)
								log('imitation_reward', new_rewards_sum)

				# reset env
				time_steps = list()
				observations = list()
				pixels = list()
				actions = list()

				time_step = self.train_env.reset()
				time_steps.append(time_step)
				actions.append(time_step.action)
				self.train_video_recorder.init(time_step.observation['pixels'])
				# try to save snapshot
				if self.cfg.save_model:
					self.save_snapshot()
				episode_step = 0
				episode_reward = 0

			# try to evaluate
			if eval_every_step(self.global_step):
				self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
				self.eval()
				
			# sample action
			with torch.no_grad(), utils.eval_mode(self.agent):
				action = self.agent.act(time_step.observation[self.cfg.obs_type], self.global_step, eval_mode=False)

			# try to update the agent
			if not seed_until_step(self.global_step):
				# Update
				metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self.global_step)
				self.logger.log_metrics(metrics, self.global_frame, ty='train')

			# take env step
			time_step = self.train_env.step(action)
			episode_reward += time_step.reward

			time_steps.append(time_step)
			observations.append(time_step.observation[self.cfg.obs_type])
			pixels.append(time_step.observation['pixels_large'])
			actions.append(time_step.action)

			self.train_video_recorder.record(time_step.observation['pixels'])
			episode_step += 1
			self._global_step += 1

	def save_snapshot(self, save_dir=None):
		snapshot = self.work_dir / 'snapshot.pt'
		if save_dir is not None:
			snapshot = save_dir
		keys_to_save = ['timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)

	def load_snapshot(self, snapshot):
		# Warning: The replay buffer is not loaded.
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
	from train import WorkspaceIL as W
	root_dir = Path.cwd()
	workspace = W(cfg)
	
	# Load weights
	if cfg.load_checkpoint:
		snapshot = Path(cfg.checkpoint_path)
		if snapshot.exists():
			print(f'resuming checkpoint: {snapshot}')
			workspace.load_snapshot(snapshot)

	workspace.train_il()

	# remove *.npz files
	if not cfg.save_experiences:
		remove_dir = workspace.work_dir / 'buffer'
		for fn in remove_dir.glob('*.npz'):
			os.remove(fn)


if __name__ == '__main__':
	main()
