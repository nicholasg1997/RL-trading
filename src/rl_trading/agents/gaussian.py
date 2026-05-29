from torch import nn
import torch

class BaseActor(nn.Module):
	def forward(self, x, deterministic=False):
		raise NotImplementedError

class GaussianActor(BaseActor):
	def __init__(self,
	             input_dim: int,
	             output_dim: int,
	             hidden_dims: tuple[int, ...] = (256, 256),
	             log_std_min: float = -20.0,
	             log_std_max: float = 2.0):
		super().__init__()
		if not hidden_dims:
			raise ValueError("hidden_dims must contain at least one layer width")

		layers = []
		last_dim = input_dim
		for hidden_dim in hidden_dims:
			layers.append(nn.Linear(last_dim, hidden_dim))
			layers.append(nn.ReLU())
			last_dim = hidden_dim

		self.net = nn.Sequential(*layers)
		self.mean_linear = nn.Linear(last_dim, output_dim)
		self.log_std_linear = nn.Linear(last_dim, output_dim)
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

	def forward(self, x, deterministic=False):
		x = self.net(x)
		mean = self.mean_linear(x)

		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		std = torch.exp(log_std)

		if deterministic:
			action = torch.tanh(mean)
			return action, None

		epsilons = torch.randn_like(mean)
		raw_action = mean + std * epsilons
		action = torch.tanh(raw_action)

		distribution = torch.distributions.Normal(mean, std)
		gaussian_log_prob = distribution.log_prob(raw_action)
		log_prob = gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6)
		log_prob = log_prob.sum(1, keepdim=True)

		return action, log_prob
