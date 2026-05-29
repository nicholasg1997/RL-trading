from torch import nn
import torch

class QNetwork(nn.Module):
	def __init__(self, obs_size: int, action_size: int, hidden_dims: tuple[int, ...] = (256, 256)):
		super().__init__()
		if not hidden_dims:
			raise ValueError("hidden_dims must contain at least one layer width")

		self.total_dims = obs_size + action_size
		layers = []
		last_dim = self.total_dims
		for hidden_dim in hidden_dims:
			layers.append(nn.Linear(last_dim, hidden_dim))
			layers.append(nn.ReLU())
			last_dim = hidden_dim

		self.net = nn.Sequential(*layers)
		self.output = nn.Linear(last_dim, 1)

	def forward(self, obs, act):
		x = torch.cat([obs, act], dim=-1)
		x = self.net(x)
		q_value = self.output(x)
		return q_value

class DoubleQCritic(nn.Module):
	def __init__(self, obs_size: int, action_size: int, hidden_dims: tuple[int, ...] = (256, 256)):
		super().__init__()
		self.q1 = QNetwork(obs_size, action_size, hidden_dims)
		self.q2 = QNetwork(obs_size, action_size, hidden_dims)

	def forward(self, obs, act):
		q1_out = self.q1(obs, act)
		q2_out = self.q2(obs, act)
		return q1_out, q2_out
