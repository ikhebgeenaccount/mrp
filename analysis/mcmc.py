import numpy as np
import scipy
from analysis.emulator import Emulator

PARAM_RANGES = {
	'Omega_m': [0.1, 0.55], 
	'S_8': [0.6, 0.9],
	'h': [0.6, 0.9],
	'w_0': [-2.0, 0.5]
}

class MCMC:

	def __init__(self, emulator: Emulator, data_vector, covariance_matrix):
		print('Instantiating MCMC')
		self.emulator = emulator
		self.data_vector = data_vector

		self.covariance_matrix = covariance_matrix
		print('Covariance matrix inversion')
		self.inv_cov_matrix = np.linalg.inv(covariance_matrix)
		print(self.inv_cov_matrix)
		
		# self.cov_matrix_det = np.linalg.det(covariance_matrix)
		# self.cov_det_inv_sqrt = 1. / np.sqrt(self.cov_matrix_det)

		self.n_slics = len(emulator.compressor.slics_pds)

		# p = len(data_vector)
		# self.c_p = scipy.special.gamma(self.n_slics / 2.) / (np.power(np.pi * (self.n_slics - 1), p/2.) * scipy.special.gamma((self.n_slics - p) / 2.))

	def log_prior(self, cosm_params):
		# Parameter ranges
		# Omega_m = 0.1 - 0.55
		# h = 0.6 - 0.82
		# S_8 = 0.6 - 0.9
		# w_0 = -2.0 - 0.5

		for param, val in zip(['Omega_m', 'S_8', 'h', 'w_0'], cosm_params):
			if val < PARAM_RANGES[param][0] or val > PARAM_RANGES[param][1]:
				return -np.inf

		return 0.
	
	def sellentin_heavens_likelihood(self, cosm_params):
		chisq = self.chi_squared(cosm_params)

		print('params=', cosm_params)
		print('chisq=', chisq)
		print('log_prior=', self.log_prior(cosm_params))
		# FIXME: negative chisq leads to nan llhood
		print('llhood=', -.5 * self.n_slics * np.log(1 + chisq / (self.n_slics + 1)) + self.log_prior(cosm_params))

		# return self.c_p * self.cov_det_inv_sqrt / np.power(1. + gauss_ll / (self.number_of_simulations - 1), self.number_of_simulations / 2.)
		return -.5 * self.n_slics * np.log(1 + chisq / (self.n_slics + 1)) + self.log_prior(cosm_params) #-.5 * np.log(self.cov_matrix_det) 

	def gaussian_likelihood(self, cosm_params):
		
		return -.5 * np.log(self.chi_squared(cosm_params)) + self.log_prior(cosm_params)

	def chi_squared(self, cosm_params):
		test_data_vector = self.emulator.predict([cosm_params])

		print('test_data_vector=', test_data_vector)
		print('data_vector=', self.data_vector)
		print('sub=', self.data_vector - test_data_vector)

		intermed = np.matmul((self.data_vector - test_data_vector), self.inv_cov_matrix)

		print('intermed=', intermed)

		return np.matmul(intermed, (self.data_vector - test_data_vector).T).flatten()
		# return -.5 * np.sum(np.square() / np.square(self.data_vector_err)) + self.log_prior(cosm_params)

	def get_random_init_walkers(self, nwalkers):
		param_rand = np.zeros(shape=(nwalkers, 4))
		for i, key in enumerate(PARAM_RANGES):

			param_rand[:, i] = np.random.uniform(low=PARAM_RANGES[key][0], high=PARAM_RANGES[key][1], size=nwalkers)
		
		return param_rand