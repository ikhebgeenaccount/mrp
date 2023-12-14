from typing import List

import numpy as np
import matplotlib.pyplot as plt

from analysis.persistence_diagram import PersistenceDiagram


class Compressor:
	"""Compressors take a list of Persistence Diagrams and compress the scatter diagrams into data vectors.
	Subclasses  must implement _build_training_set and _build_equivalent_slics_set.
	"""

	def __init__(self, cosmoslics_pds: List[PersistenceDiagram], slics_pds: List[PersistenceDiagram]):
		self.cosmoslics_pds = cosmoslics_pds
		self.slics_pds = slics_pds

		self.cosmoslics_training_set = self._build_training_set(cosmoslics_pds)
		self.slics_training_set = self._build_training_set(slics_pds)
		self._build_covariance_matrix()

	def _build_training_set(self, pds: List[PersistenceDiagram]):
		"""Build the training set to be used with an Emulator. Must return the training set.
		Training sets are dictionaries containing name, input and target fields.		
		"""
		raise NotImplementedError

	def _build_crosscorr_matrix(self):
		self.slics_crosscorr_matrix = np.corrcoef(self.slics_training_set['target'].T)
		# self.cosmoslics_covariance_matrix = np.corrcoef(self.cosmoslics_training_set['target'].T)

	def _build_covariance_matrix(self):
		self.slics_covariance_matrix = np.cov(self.slics_training_set['target'])

	def _plot_matrix(self, matrix, title=''):
		fig, ax = plt.subplots()
		imax = ax.imshow(matrix)
		fig.colorbar(imax)
		ax.set_title(title)
		return fig, ax

	def plot_covariance_matrices(self):
		self._plot_matrix(self.slics_covariance_matrix, 'SLICS covariance matrix')
		# self._plot_cov_matrix(self.cosmoslics_covariance_matrix, 'cosmoSLICS covariance matrix')
	
	def plot_crosscorr_matrix(self):
		self._build_crosscorr_matrix()
		self._plot_matrix(self.slics_crosscorr_matrix, 'SLICS crosscorr matrix')