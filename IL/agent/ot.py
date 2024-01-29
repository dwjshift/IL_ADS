import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import matplotlib.pyplot as plt
import pickle

import utils
from agent.encoder import Encoder
from rewarder import optimal_transport_plan, cosine_distance, euclidean_distance
import time
import copy

class Actor(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, hidden_dim),
									nn.ReLU(inplace=True),
									nn.Linear(hidden_dim, action_shape[0]))

		self.apply(utils.weight_init)

	def forward(self, obs, std):
		h = self.trunk(obs)
		mu = self.policy(h)
		mu = torch.tanh(mu)
		std = torch.ones_like(mu) * std

		dist = utils.TruncatedNormal(mu, std)
		return dist


class Critic(nn.Module):
	def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
		super().__init__()

		self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
								   nn.LayerNorm(feature_dim), nn.Tanh())

		self.Q1 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.Q2 = nn.Sequential(
			nn.Linear(feature_dim + action_shape[0], hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

		self.apply(utils.weight_init)

	def forward(self, obs, action):
		h = self.trunk(obs)
		h_action = torch.cat([h, action], dim=-1)
		q1 = self.Q1(h_action)
		q2 = self.Q2(h_action)

		return q1, q2


class OTAgent:
	def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
				 hidden_dim, critic_target_tau, num_expl_steps,
				 update_every_steps, stddev_schedule, stddev_clip, use_tb, augment,
				 rewards, sinkhorn_rew_scale, update_target_every,
				 auto_rew_scale, auto_rew_scale_factor, suite_name, name, obs_type, update_policy_freq, expl_noise):
		self.device = device
		self.lr = lr
		self.critic_target_tau = critic_target_tau
		self.update_every_steps = update_every_steps
		self.use_tb = use_tb
		self.num_expl_steps = num_expl_steps
		self.stddev_schedule = stddev_schedule
		self.stddev_clip = stddev_clip
		self.update_policy_freq = update_policy_freq
		self.expl_noise = expl_noise
		self.augment = augment
		self.rewards = rewards
		self.sinkhorn_rew_scale = sinkhorn_rew_scale
		self.update_target_every = update_target_every
		self.auto_rew_scale = auto_rew_scale
		self.auto_rew_scale_factor = auto_rew_scale_factor
		self.use_encoder = True if obs_type=='pixels' else False
		self.obs_type = obs_type

		# models
		if self.use_encoder:
			self.encoder = Encoder(obs_shape).to(device)
			self.encoder_target = Encoder(obs_shape).to(device)
			repr_dim = self.encoder.repr_dim
		else:
			repr_dim = obs_shape[0]

		self.actor = Actor(repr_dim, action_shape, feature_dim,
						   hidden_dim).to(device)

		self.critic = Critic(repr_dim, action_shape, feature_dim,
							 hidden_dim).to(device)
		self.critic_target = Critic(repr_dim, action_shape,
									feature_dim, hidden_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

		# data augmentation
		self.aug = utils.RandomShiftsAug(pad=4)

		self.train()
		self.critic_target.train()

	def __repr__(self):
		return "ot"

	def train(self, training=True):
		self.training = training
		if self.use_encoder:
			self.encoder.train(training)
		self.actor.train(training)
		self.critic.train(training)

	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)
		obs = self.encoder(obs.unsqueeze(0)) if self.use_encoder else obs.unsqueeze(0)

		stddev = utils.schedule(self.expl_noise, step)
		dist = self.actor(obs, stddev)

		if eval_mode:
			action = dist.mean
		else:
			action = dist.sample(clip=None)
			if step < self.num_expl_steps:
				action.uniform_(-1.0, 1.0)
		return action.cpu().numpy()[0]

	def update_critic(self, obs, action, reward, discount, next_obs, step):
		metrics = dict()

		with torch.no_grad():
			stddev = utils.schedule(self.stddev_schedule, step)
			dist = self.actor(next_obs, stddev)
			next_action = dist.sample(clip=self.stddev_clip)
			target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
			target_V = torch.min(target_Q1, target_Q2)
			target_Q = reward + (discount * target_V)

		Q1, Q2 = self.critic(obs, action)

		critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

		# optimize encoder and critic
		if self.use_encoder:
			self.encoder_opt.zero_grad(set_to_none=True)
		self.critic_opt.zero_grad(set_to_none=True)
		critic_loss.backward()
		self.critic_opt.step()
		if self.use_encoder:
			self.encoder_opt.step()

		if self.use_tb:
			metrics['critic_target_q'] = target_Q.mean().item()
			metrics['critic_q1'] = Q1.mean().item()
			metrics['critic_q2'] = Q2.mean().item()
			metrics['critic_loss'] = critic_loss.item()
			
		return metrics

	def update_actor(self, obs, step):
		metrics = dict()

		stddev = utils.schedule(self.stddev_schedule, step)

		dist = self.actor(obs, stddev)
		action = dist.sample(clip=self.stddev_clip)
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)

		Q1, Q2 = self.critic(obs, action)
		Q = torch.min(Q1, Q2)
		actor_loss = - Q.mean()

		# optimize actor
		self.actor_opt.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.actor_opt.step()
		if self.use_tb:
			metrics['actor_loss'] = actor_loss.item()
			metrics['actor_logprob'] = log_prob.mean().item()
			metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
			metrics['actor_q'] = Q.mean().item()
			metrics['rl_loss'] = -Q.mean().item()
		return metrics

	def update(self, replay_iter, expert_replay_iter, step):
		metrics = dict()

		if step % self.update_every_steps != 0:
			return metrics

		batch = next(replay_iter)
		obs, action, reward, discount, next_obs = utils.to_torch(
			batch, self.device)

		# augment
		if self.use_encoder and self.augment:
			obs = self.aug(obs.float())
			next_obs = self.aug(next_obs.float())
		else:
			obs = obs.float()
			next_obs = next_obs.float()

		if self.use_encoder:
			# encode
			obs = self.encoder(obs)
			with torch.no_grad():
				next_obs = self.encoder(next_obs)

		if self.use_tb:
			metrics['batch_reward'] = reward.mean().item()

		# update critic
		metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

		# update actor
		if step % (self.update_every_steps * self.update_policy_freq) == 0:
			metrics.update(self.update_actor(obs.detach(), step))

		# update critic target
		utils.soft_update_params(self.critic, self.critic_target,
								 self.critic_target_tau)

		return metrics

	def init_demos(self, demos, cost_encoder):
		self.cost_encoder = cost_encoder
		self.demos = demos

	def ot_rewarder(self, observations, step, return_infos=False):
		scores_list = list()
		ot_rewards_list = list()
		if return_infos:
			cost_matrix_list = list()
			transport_plan_list = list()

		obs = torch.tensor(observations).to(self.device).float()
		obs = obs.detach()
		if self.use_encoder:
			with torch.no_grad():
				obs = self.cost_encoder(obs)
		
		for demo in self.demos:
			exp = torch.tensor(demo).to(self.device).float()
			exp = exp.detach()
			assert obs.shape == exp.shape
			
			if self.rewards == 'sinkhorn_cosine':
				cost_matrix = cosine_distance(obs, exp)
			elif self.rewards == 'sinkhorn_euclidean':
				cost_matrix = euclidean_distance(obs, exp)
			else:
				raise NotImplementedError()

			if return_infos:
				cost_matrix_list.append(cost_matrix.detach().cpu().numpy())
			transport_plan = optimal_transport_plan(
				obs, exp, cost_matrix, method='sinkhorn',
				niter=100).float()  # Getting optimal coupling
			ot_rewards = -self.sinkhorn_rew_scale * torch.diag(
				torch.mm(transport_plan, cost_matrix.T)).detach().cpu().numpy()

			if return_infos:
				transport_plan_list.append(transport_plan.detach().cpu().numpy())
			scores_list.append(np.sum(ot_rewards) / self.sinkhorn_rew_scale)
			ot_rewards_list.append(ot_rewards)

		closest_demo_index = np.argmax(scores_list)
		if return_infos:
			return cost_matrix_list[closest_demo_index], transport_plan_list[closest_demo_index], ot_rewards_list[closest_demo_index], np.max(scores_list)
		return ot_rewards_list[closest_demo_index]

	def save_snapshot(self):
		keys_to_save = ['actor', 'critic', 'actor_opt', 'critic_opt', 'sinkhorn_rew_scale']
		if self.use_encoder:
			keys_to_save += ['encoder', 'encoder_opt']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		return payload

	def load_snapshot(self, payload):
		for k, v in payload.items():
			self.__dict__[k] = v
		self.critic_target.load_state_dict(self.critic.state_dict())
		if self.use_encoder:
			self.encoder_target.load_state_dict(self.encoder.state_dict())

		# Update optimizers
		if self.use_encoder:
			self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
			self.encoder_opt.load_state_dict(payload['encoder_opt'].state_dict())
		self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.actor_opt.load_state_dict(payload['actor_opt'].state_dict())
		self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
		self.critic_opt.load_state_dict(payload['critic_opt'].state_dict())

		print('sinkhorn_rew_scale of the loaded model:', payload['sinkhorn_rew_scale'])
