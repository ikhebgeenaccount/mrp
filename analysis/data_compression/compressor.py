from typing import List
import re

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

from analysis.cosmology_data import CosmologyData
from analysis.persistence_diagram import PersistenceDiagram
from utils.file_system import check_folder_exists


class Compressor:
	"""Compressors take a list of Persistence Diagrams and compress the scatter diagrams into data vectors.
	Subclasses  must implement _build_training_set and _build_equivalent_slics_set.
	"""

	def __init__(self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData]):
		self.cosmoslics_datas = cosmoslics_datas
		self.slics_data = slics_data

		self.zbins = slics_data[0].zbins
		pat = re.compile('([0-9.]+)')
		self.zbins_labels = []
		for zbin in self.zbins:
			det = pat.findall(zbin)
			
			lbl = ''
			if len(det) == 2:
				lbl = '{0} < $z$ < {1}'.format(*det)
			else:
				lbl = '{0} < $z$ < {1} X {2} < $z$ < {3}'.format(*det)
			
			self.zbins_labels.append(lbl)

		self.plots_dir = 'plots'

	def compress(self):
		self.cosmoslics_training_set = self._build_training_set(self.cosmoslics_datas)
		self.cosmoslics_training_set['target'] = np.array(self.cosmoslics_training_set['target'])
		
		self.input_vector_length = len(self.cosmoslics_training_set['input'][0])
		self.data_vector_length = len(self.cosmoslics_training_set['target'][0])

		self.slics_training_set = self._build_slics_training_set(self.slics_data)
		self.slics_training_set['target'] = np.array(self.slics_training_set['target'])
		self._build_covariance_matrix()
		self._calculate_average_data_vector()

		self._calculate_derivatives_lsq()
		self._calculate_fisher_matrix()

	def _build_training_set(self, cosm_datas: List[CosmologyData]):
		"""Build the training set to be used with an Emulator. Must return the training set.
		Training sets are dictionaries containing name, input and target fields.		
		"""
		raise NotImplementedError

	def _build_slics_training_set(self, slics_data: List[CosmologyData]):
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
		# self.slics_covariance_matrix = self.slics_covariance_matrix / len(self.cosmoslics_datas[0].zbins_pds[self.zbins[0]])

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
		# self.fisher_matrix_per_entry = np.zeros((self.input_vector_length, self.input_vector_length, self.data_vector_length))
		slics_variance = np.diag(self.slics_covariance_matrix)
		
		for i in range(self.input_vector_length):
			for j in range(self.input_vector_length):
				# TODO: add variance
				s = self.lsq_sols[:, i] * self.lsq_sols[:, j] / slics_variance
				# self.fisher_matrix_per_entry[j, i] = s
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

	def _plot_matrix(self, matrix, title='', origin=None, save_name=None, save=True):
		fig, ax = plt.subplots()
		imax = ax.imshow(matrix, origin=origin)
		fig.colorbar(imax)
		ax.set_title(title)

		if save_name is not None and save:
			self._save_plot(fig, save_name)
		return fig, ax
	
	def _save_plot(self, fig, save_name):
		check_folder_exists(self.plots_dir)
		fig.tight_layout()
		fig.savefig(f'{self.plots_dir}/{type(self).__name__}_{save_name}.pdf')
		plt.close(fig)

	def plot_covariance_matrix(self, save=True):
		self._plot_matrix(self.slics_covariance_matrix, title='SLICS covariance matrix', save_name='slics_cov_matrix', save=save)
		# self._plot_cov_matrix(self.cosmoslics_covariance_matrix, 'cosmoSLICS covariance matrix')
	
	def plot_correlation_matrix(self, save=True):
		self._build_crosscorr_matrix()
		self._plot_matrix(self.slics_crosscorr_matrix, title='SLICS correlation matrix', save_name='slics_corr_matrix', save=save)

	def plot_data_vectors(self, include_slics=False, include_cosmoslics=True, save=True, true_value=False, logy=False, entries_per_row=50):
		if true_value:
			cosmoslics_plot = self.cosmoslics_training_set['target']
			slics_plot = self.avg_slics_data_vector
			slics_err_plot = self.avg_slics_data_vector_err
		else:
			# Normalize by dividing by average of cosmoSLICS data vectors
			cosmoslics_plot = (self.cosmoslics_training_set['target'] - self.avg_slics_data_vector) / (self.avg_slics_data_vector_err)
			slics_plot = np.zeros(self.data_vector_length)
			slics_err_plot = np.ones(self.data_vector_length)

		if logy:
			cosmoslics_plot = np.abs(cosmoslics_plot)
			slics_plot = np.abs(slics_plot)
			slics_err_plot = np.abs(slics_err_plot)

		if self.data_vector_length > entries_per_row:
			nrows = int(np.ceil(self.data_vector_length / 50))
		else:
			nrows = 1

		# Width is increased with longer data vector
		fig, axs = plt.subplots(nrows=nrows, figsize=(max(6.4, 6.4 / 35 * self.data_vector_length / nrows), 3 * nrows), sharey=True)

		for row_id in range(nrows):
			if nrows == 1:
				ax = axs
			else:
				ax = axs[row_id]

			vector_slice = np.s_[row_id * 50:min(self.data_vector_length, (row_id +1) * 50)]
			x = np.arange(vector_slice.start, vector_slice.stop)

			if include_cosmoslics:
				ax.plot(x, cosmoslics_plot.T[vector_slice], color='blue', alpha=.4, linewidth=1)
				# Empty plot for legend
				ax.plot(np.nan, color='blue', alpha=.4, linewidth=1, label='cosmoSLICS')

			if include_slics:
				ax.plot(x, slics_plot[vector_slice], color='red', linewidth=3, alpha=.6, label='SLICS')

				ax.fill_between(
					x,
					y1=slics_plot[vector_slice] + slics_err_plot[vector_slice],
					y2=slics_plot[vector_slice] - slics_err_plot[vector_slice],
					color='grey', alpha=.4, label='$1\sigma$ SLICS covariance'
				)

			if logy:
				ax.semilogy()

			if true_value:
				# ax.set_title('Data vectors')
				y_label = 'Entry value'
			else:
				# ax.set_title('Data vectors normalized with cosmoSLICS average')
				y_label = '(cosmoSLICS - SLICS) / $\sigma_\mathrm{SLICS}$'

			if logy:
				y_label = f'|{y_label}|'
		
			ax.set_ylabel(y_label)

		ax.set_xlabel('Data vector entry')	
		ax.legend()				

		if save:
			fig.tight_layout()
			self._save_plot(fig, f'data_vector{"" if not true_value else "_abs"}{"" if not logy else "_logy"}')

		return fig, axs
	
	def plot_fisher_matrix(self, save=True):
		fig, ax = self._plot_matrix(self.fisher_matrix, title='Fisher information matrix')

		ax.set_xticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])
		ax.set_yticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])

		if save:
			self._save_plot(fig, 'fisher_matrix')
		
		fig, ax = self._plot_matrix(self.fisher_corr_matrix, title='Fisher information correlation matrix')

		ax.set_xticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])
		ax.set_yticks(ticks=[0, 1, 2, 3], labels=['$\Omega_m$', '$S_8$', '$h$', '$w_0$'])

		if save:
			self._save_plot(fig, 'fisher_corr_matrix')

	def debug(self, message):
		if self.verbose:
			print(message)