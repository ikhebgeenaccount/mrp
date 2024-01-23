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

	def log_likelihood(self, cosm_params):
		test_data_vector = self.emulator.predict([cosm_params])

		return -.5 * np.sum(np.square(self.data_vector - test_data_vector) / self.data_vector_err)