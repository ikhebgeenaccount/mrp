from matplotlib import pyplot as plt
import numpy as np
from analysis.data_compression.compressor import Compressor
from analysis.data_compression.criteria.criterium import Criterium


class ChiSquared(Criterium):

	def __init__(self, slics_data, dist_powers, chisq_increase):	
		self.slics_data = slics_data
		self.dist_powers = dist_powers
		self.chisq_increase = chisq_increase
		self.prev_chisq = 0.
		self.chisq_values = []
		self.fisher_dets = []

	def acceptance_func(self, compressor: Compressor):
		# Calculate chi squared value
		chi_sq = self.criterium_value(compressor)
		fisher_det = np.linalg.det(compressor.fisher_matrix)

		if chi_sq - self.prev_chisq > self.chisq_increase:
			self.prev_chisq = chi_sq

			self.chisq_values.append(chi_sq)
			self.fisher_dets.append(fisher_det)

			return True
		return False
	
	def criterium_value(self, compressor: Compressor):
		sub = (compressor.avg_slics_data_vector - compressor.cosmoslics_training_set['target'])
		intermed = np.matmul(sub, np.linalg.inv(compressor.slics_covariance_matrix))
		return (1. / 26.) * np.sum(np.matmul(intermed, sub.T))
	
	def pixel_scores(self):
		dp_merge = []
		for zbin in self.slics_data[0].zbins:
			dp_merge.append([self.dist_powers[zbin][0]._transform_map(), self.dist_powers[zbin][1]._transform_map()])
		return np.array(dp_merge)
	
	def plot(self):		
		fig, ax = plt.subplots()
		ax.set_ylabel('$\chi^2$')
		ax.set_xlabel('Data vector entry')
		ax.plot(self.chisq_values)
		return fig
	