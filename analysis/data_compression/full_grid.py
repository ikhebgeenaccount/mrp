from typing import List

import numpy as np

from analysis.cosmology_data import CosmologyData
from analysis.data_compression.index_compressor import IndexCompressor
from analysis.persistence_diagram import PersistenceDiagram


class FullGrid(IndexCompressor):

	def __init__(self, cosmoslics_datas: List[CosmologyData], slics_data: List[CosmologyData]):
		# Build set of indices that contain all entries
		indices = np.indices((len(slics_data[0].zbins_pds), 2, 100, 100))
		super().__init__(cosmoslics_datas, slics_data, indices=indices.T)

	def _calculate_fisher_matrix(self):
		self.fisher_matrix = np.zeros((self.input_vector_length, self.input_vector_length))
		# Add fisher_matrix_per_entry values, as this is what FisherInfoMaximizer uses to determine best pixels
		self.fisher_matrix_per_entry = np.zeros((self.input_vector_length, self.input_vector_length, self.data_vector_length))
		slics_variance = np.square(self.avg_slics_data_vector_err)
		
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
	
	def _build_covariance_matrix(self):
		# Building a covariance matrix for FullGrid is not possible, as it has 16(zbins) x 20000(bngs) = 320k data points
		self.slics_covariance_matrix = np.array([[1.]])  # Set it to None so an error is thrown when it is used

	# def _calculate_average_data_vector(self):
	# 	if self.data_vector_length == 1:
	# 		self.avg_slics_data_vector = np.array([np.average(self.slics_training_set['target'])])
	# 		self.avg_cosmoslics_data_vector = np.array([np.average(self.cosmoslics_training_set['target'])])
	# 	else:
	# 		self.avg_slics_data_vector = np.average(self.slics_training_set['target'], axis=0)
	# 		self.avg_cosmoslics_data_vector = np.average(self.cosmoslics_training_set['target'], axis=0)
	# 	self.avg_slics_data_vector_err = np.std(self.slics_training_set['target'], axis=0)