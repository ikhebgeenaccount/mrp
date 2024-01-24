import numpy as np
from analysis.emulator import Emulator

class MCMC:

	def __init__(self, emulator: Emulator, data_vector, data_vector_err=None):
		self.emulator = emulator
		self.data_vector = data_vector

		if data_vector_err is None:
			self.data_vector_err = np.ones(self.data_vector.shape)
		else:
			self.data_vector_err = data_vector_err

	def log_prior(self, cosm_params):
		# Parameter ranges
		# Omega_m = 0.1 - 0.55
		# h = 0.6 - 0.82
		# S_8 = 0.6 - 0.9
		# w = -2.0 - 0.5

		if np.any(cosm_params < 0.) or np.any(cosm_params > 5):
			return -np.inf

		return 0.


	def log_likelihood(self, cosm_params):
		test_data_vector = self.emulator.predict([cosm_params])

		return -.5 * np.sum(np.square(self.data_vector - test_data_vector) / self.data_vector_err) + self.log_prior(cosm_params)