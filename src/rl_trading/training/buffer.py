from abc import ABC
from ..configs.data_classes import Transition
import torch

import numpy as np


class Buffer(ABC):
	"Base class to represent buffers for off-policy algorithms."

	def __init__(self, buffer_size: int, obs_space, act_space, device):
		super().__init__()
		self.buffer_size = buffer_size
		self.obs_space = obs_space
		self.act_space = act_space
		self.device = device

		# circular buffer
		self.pointer = 0
		self.full = False


	def add(self, *args, **kwargs):
		"Add elements to the buffer."
		raise NotImplementedError

	def reset(self):
		"Reset the buffer, i.e. empty all its contents."
		raise NotImplementedError

	def sample(self, batch_size: int):
		"Sample a batch of elements from the buffer."
		raise NotImplementedError

	def _get_samples(self, indices):
		"Return a batch of elements from the buffer."
		raise NotImplementedError

	def __len__(self):
		"Return the current size of the buffer."
		if self.full:
			return self.buffer_size
		else:
			return self.pointer

class ReplayBuffer(Buffer):
	def __init__(self,
	             buffer_size,
	             obs_space,
	             act_space,
	             device):
		super().__init__(buffer_size, obs_space, act_space, device)
		self.buffer_size = buffer_size
		self.device = device

		self.obs = np.zeros((buffer_size, *obs_space.shape), dtype=np.float32)
		self.act = np.zeros((buffer_size, *act_space.shape), dtype=np.float32)
		self.rew = np.zeros(buffer_size, dtype=np.float32)
		self.next_obs = np.zeros((buffer_size, *obs_space.shape), dtype=np.float32)
		self.terminated = np.zeros(buffer_size, dtype=np.float32)
		self.truncated = np.zeros(buffer_size, dtype=np.float32)

	def reset(self):
		self.pointer = 0
		self.full = False

	def add(self, obs, act, rew, next_obs, terminated, truncated):
		self.obs[self.pointer] = obs
		self.act[self.pointer] = act
		self.rew[self.pointer] = rew
		self.next_obs[self.pointer] = next_obs
		self.terminated[self.pointer] = terminated
		self.truncated[self.pointer] = truncated

		self.pointer = (self.pointer + 1) % self.buffer_size
		if self.pointer == 0:
			self.full = True

	def sample(self, batch_size: int):
		indices = np.random.randint(0, len(self), batch_size)
		return self._get_samples(indices)

	def _get_samples(self, indices):
		return Transition(
			obs = torch.FloatTensor(self.obs[indices]).to(self.device),
			act = torch.FloatTensor(self.act[indices]).to(self.device),
			rew = torch.FloatTensor(self.rew[indices]).unsqueeze(-1).to(self.device),
			next_obs = torch.FloatTensor(self.next_obs[indices]).to(self.device),
			terminated=torch.FloatTensor(self.terminated[indices]).unsqueeze(-1).to(self.device),
			truncated=torch.FloatTensor(self.truncated[indices]).unsqueeze(-1).to(self.device),
		)
