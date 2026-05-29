from abc import ABC


class BaseLogger(ABC):
	def log_scalar(self, key: str, value: float, step: int):
		"Log a scalar at a given step."
		raise NotImplementedError

	def log_dict(self, metrics: dict[str, float], step: int):
		"Log a dictionary of scalars at a given step."
		raise NotImplementedError

	def close(self):
		raise NotImplementedError

class CompositeLogger(BaseLogger):
	def __init__(self, loggers: list[BaseLogger]):
		self.loggers = loggers

	def log_scalar(self, key: str, value: float, step: int):
		for logger in self.loggers:
				logger.log_scalar(key, value, step)

	def log_dict(self, metrics: dict[str, float], step: int):
		for logger in self.loggers:
				logger.log_dict(metrics, step)

	def close(self):
		for logger in self.loggers:
				logger.close()
