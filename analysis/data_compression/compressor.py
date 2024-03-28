from typing import List

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

from analysis.persistence_diagram import PersistenceDiagram


class Compressor:
	"""Compressors take a list of Persistence Diagrams and compress the scatter diagrams into data vectors.
	Subclasses  must implement _build_training_set and _build_equivalent_slics_set.
	"""

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram]):
		self.cosmoslics_pds = cosmoslics_pds
		self.slics_pds = slics_pds

		self.cosmoslics_training_set = self._build_training_set(cosmoslics_pds)
		self.cosmoslics_training_set['target'] = np.array(self.cosmoslics_training_set['target'])
		
		self.input_vector_length = len(self.cosmoslics_training_set['input'][0])
		self.data_vector_length = len(self.cosmoslics_training_set['target'][0])

		self.slics_training_set = self._build_training_set(slics_pds)
		self.slics_training_set['target'] = np.array(self.slics_training_set['target'])
		self._build_covariance_matrix()
		self._calculate_average_data_vector()

		self._calculate_derivatives_lsq()
		self._calculate_fisher_matrix()

		self.plots_dir = 'plots'

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		"""Build the training set to be used with an Emulator. Must return the training set.
		Training sets are dictionaries containing name, input and target fields.		
		"""
		raise NotImplementedError

	def _build_crosscorr_matrix(self):
		if self.data_vector_length == 1:
			self.slics_crosscorr_matrix = np.array([[1.]])
		else:
			self.slics_crosscorr_matrix = np.corrcoef(self.slics_training_set['target'].T)
		# self.cosmoslics_covariance_matrix = np.corrcoef(self.cosmoslics_training_set['target'].T)

	def _build_covariance_matrix(self):
		if self.data_vector_length == 1:
			self.slics_covariance_matrix = np.array([[np.var(self.slics_training_set['target'].T)]])
		else:
			self.slics_covariance_matrix = np.cov(self.slics_training_set['target'].T)

	def _calculate_average_data_vector(self):
		if self.data_vector_length == 1:
			self.avg_slics_data_vector = np.array([np.average(self.slics_training_set['target'])])
			self.avg_cosmoslics_data_vector = np.array([np.average(self.cosmoslics_training_set['target'])])
		else:
			self.avg_slics_data_vector = np.average(self.slics_training_set['target'], axis=0)
			self.avg_cosmoslics_data_vector = np.average(self.cosmoslics_training_set['target'], axis=0)
		self.avg_slics_data_vector_err = np.sqrt(np.diag(self.slics_covariance_matrix))

	def _calculate_derivatives_lsq(self):
		# lsq_linear minimizes system of equations
		# Ax - b 
		# A are the cosmological parameters
		# x are the functional parameters
		# b are the pixel values corresponding to A's params

		# TODO: lsq with variance?
		self.lsq_sols = np.zeros((self.data_vector_length, self.input_vector_length))
		
		for entry in range(self.data_vector_length):
			# Delta with SLICS
			A = self.cosmoslics_training_set['input'] - self.slics_training_set['input'][0]
			b = self.cosmoslics_training_set['target'][:, entry] - self.avg_slics_data_vector[entry]

			res = lsq_linear(A, b)

			self.lsq_sols[entry] = res.x

		return self.lsq_sols
	
	def _calculate_fisher_matrix(self):
		self.fisher_matrix = np.zeros((self.input_vector_length, self.input_vector_length))
		self.fisher_matrix_per_entry = np.zeros((self.input_vector_length, self.input_vector_length, self.data_vector_length))
		slics_variance = np.diag(self.slics_covariance_matrix)
		
		for i in range(self.input_vector_length):
			for j in range(self.input_vector_length):
				# TODO: add variance
				s = self.lsq_sols[:, i] * self.lsq_sols[:, j] / slics_variance
				self.fisher_matrix_per_entry[j, i] = s
				self.fisher_matrix[j, i] = np.sum(s[np.isfinite(s)])
				# self.fisher_matrix[j, i] = np.sum(self.lsq_sols[:, i] * self.lsq_sols[:, j])

		# Calculate Fisher correlation matrix
		# F_ij / sqrt(F_ii * F_jj)
		diag = np.sqrt(np.diag(self.fisher_matrix))
		self.fisher_corr_matrix = self.fisher_matrix / np.outer(diag, diag)		

	def _calculate_derivatives_odr(self):
		from scipy.odr import Model, Data, ODR

		# Function to be fit, is linear in all parameters
		def f(B, x):
			# x are the values of the cosm parameters
			# B are the parameters
			return B[0] + np.sum(B[1:] * x)

		for entry in range(self.data_vector_length):

			linear = Model(f)
			dat = Data(self.cosmoslics_training_set['input'], self.cosmoslics_training_set['target'][:, entry])
			odr_ = ODR(dat, linear, beta0=np.ones(5))

			out = odr_.run()
			out.pprint()

	def _plot_matrix(self, matrix, title='', origin=None, save_name=None):
		fig, ax = plt.subplots()
		imax = ax.imshow(matrix, origin=origin)
		fig.colorbar(imax)
		ax.set_title(title)

		if save_name is not None:
			self._save_plot(fig, save_name)
		return fig, ax
	
	def _save_plot(self, fig, save_name):
		fig.savefig(f'{self.plots_dir}/{type(self).__name__}_{save_name}.png')

	def plot_covariance_matrix(self):
		self._plot_matrix(self.slics_covariance_matrix, title='SLICS covariance matrix', save_name='slics_cov_matrix')
		# self._plot_cov_matrix(self.cosmoslics_covariance_matrix, 'cosmoSLICS covariance matrix')
	
	def plot_correlation_matrix(self):
		self._build_crosscorr_matrix()
		self._plot_matrix(self.slics_crosscorr_matrix, title='SLICS correlation matrix', save_name='slics_corr_matrix')

	def plot_data_vectors(self, include_slics=False, include_cosmoslics=True, save=True):
		fig, ax = plt.subplots()

		# Normalize by dividing by average of cosmoSLICS data vectors
		avg_cosmoslics_data_vector = np.average(self.cosmoslics_training_set['target'], axis=0)
		cosmoslics_plot = self.cosmoslics_training_set['target'] / avg_cosmoslics_data_vector

		if include_cosmoslics:
			ax.plot(cosmoslics_plot.T, color='blue', alpha=.4, linewidth=1)
			# Empty plot for legend
			ax.plot(np.nan, color='blue', alpha=.4, linewidth=1, label='cosmoSLICS')

		if include_slics:
			slics_norm = self.avg_slics_data_vector / avg_cosmoslics_data_vector
			ax.plot(slics_norm, color='red', linewidth=3, alpha=.6, label='SLICS')

			slics_err_norm = self.avg_slics_data_vector_err / avg_cosmoslics_data_vector

			ax.fill_between(
				np.linspace(0, self.data_vector_length - 1, num=self.data_vector_length),
				y1=slics_norm + slics_err_norm,
				y2=slics_norm - slics_err_norm,
				color='grey', alpha=.4, label='$1\sigma$ SLICS covariance'
			)

		ax.legend()
		ax.set_title('Data vectors normalized with cosmoSLICS average')
		ax.set_xlabel('Data vector entry')
		ax.set_ylabel('Entry value / cosmoSLICS avg')

		if save:
			self._save_plot(fig, 'data_vector')

		return fig, ax
	
	def plot_fisher_matrix(self):
		fig, ax = self._plot_matrix(self.fisher_matrix, title='Fisher information matrix')

		ax.set_xticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])
		ax.set_yticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])

		self._save_plot(fig, 'fisher_matrix')
		
		fig, ax = self._plot_matrix(self.fisher_corr_matrix, title='Fisher information correlation matrix')

		ax.set_xticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])
		ax.set_yticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])

		self._save_plot(fig, 'fisher_corr_matrix')

	def debug(self, message):
		if self.verbose:
			print(message)