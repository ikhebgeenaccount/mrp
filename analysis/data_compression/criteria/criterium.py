

from analysis.data_compression.compressor import Compressor


class Criterium:

	def acceptance_func(self, compressor: Compressor):
		raise NotImplementedError(f'Subclasses of {type(self).__name__} must implement acceptance_func')
	
	def criterium_value(self, compressor: Compressor):
		raise NotImplementedError(f'Subclasses of {type(self).__name__} must implement criterium')
	
	def pixel_scores(self):
		raise NotImplementedError(f'Subclasses of {type(self).__name__} must implement at least one of pixel_scores and pixel_scores_per_zbin')
	
	def pixel_scores_per_zbin(self):
		raise NotImplementedError(f'Subclasses of {type(self).__name__} must implement at least one of pixel_scores and pixel_scores_per_zbin')
	
	def plot(self):
		pass