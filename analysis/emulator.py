import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from analysis.data_compression.compressor import Compressor

class Emulator:

	def __init__(self, regressor_type, training_set=None, compressor: Compressor=None, **regressor_args):
		if training_set is not None:
			self.compressor = None
			self.training_set = training_set
		elif compressor is not None:
			self.compressor = compressor
			self.training_set = compressor.cosmoslics_training_set

		self.regressor_type = regressor_type
		self.regressor_args = regressor_args
		self.regressor = regressor_type(**regressor_args)

		self.standard_scaler = StandardScaler()
		# Ensure that training_set contains numpy.ndarrays
		self.training_set['input'] = np.array(self.training_set['input'])
		self.training_set['scaled_input'] = self.standard_scaler.fit_transform(self.training_set['input'])
		self.training_set['target'] = np.array(self.training_set['target'])
		# self.training_set['scaled_target'] = self.standard_scaler.fit_transform(self.training_set['target'])

	def fit(self):
		self.regressor.fit(self.training_set['scaled_input'], self.training_set['target'])
	
	def predict(self, X):
		# scale X through self.standard_scaler, not fitting again
		return self.regressor.predict(self.standard_scaler.transform(X))
	
	def validate(self, make_plot=False):
		loo = LeaveOneOut()

		all_mse = []

		for i, (train_index, test_index) in enumerate(loo.split(self.training_set['scaled_input'])):
			temp_regr = self.regressor_type(**self.regressor_args)
			temp_regr.fit(self.training_set['scaled_input'][train_index], self.training_set['target'][train_index])

			# Not np.abs to also get negative error
			mse = (self.training_set['target'][test_index][0] - temp_regr.predict(self.training_set['scaled_input'][test_index])[0]) / self.training_set['target'][test_index][0]
			# mse = np.square((self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['scaled_input'][test_index])[0]) / self.training_set['target'][test_index][0])

			all_mse.append(mse)

		avg_mse = np.average(all_mse, axis=0)

		if make_plot:
			self.create_loocv_plot(avg_mse, all_mse)

		return avg_mse, all_mse

	def hyperparameter_optimization(self, param_grid):
		# TODO
		rcv = RandomizedSearchCV()
		pass


	def create_loocv_plot(self, avg_mse, all_mse):
		fig, ax = plt.subplots()
		ax.set_title(f'{self.training_set["name"]} after LOOCV')
		
		ax.plot(avg_mse, label='Average fractional error', color='red', linewidth=3)

		ax.plot(np.array(all_mse).T, color='black', alpha=.2, linewidth=1)
		
		ax.legend()
		ax.set_xlabel('Data vector entry')
		ax.set_ylabel('Fractional error')
		return fig, ax


class GPREmulator(Emulator):

	def __init__(self, training_set=None, compressor=None, **regressor_args) -> None:
		super().__init__(GaussianProcessRegressor, training_set=training_set, compressor=compressor, normalize_y=True, **regressor_args)


class MLPREmulator(Emulator):

	def __init__(self, training_set) -> None:
		super().__init__(MLPRegressor, training_set)


class PerFeatureEmulator(Emulator):
	"""
	Instead of having one GPR for the whole data vector, we have one for each entry."""

	def __init__(self, training_set, regressor_type):
		super().__init__(GaussianProcessRegressor, training_set, normalize_y=True)

		# Create an emulator for each entry in the target data vector
		self.regressors = [regressor_type(**self.regressor_args) for _ in training_set['target'][0]]
	
	def fit(self):
		pass

	def predict(self, X):
		scaled_X = self.standard_scaler.transform(X)
		# TODO: fix for multiple predictions in X at once (multiple input vectors in X)
		return np.array([[regressor.fit(scaled_X).flatten()[0] for regressor in self.regressors]])

	def validate(self):
		loo = LeaveOneOut()

		all_mse = []

		for i, (train_index, test_index) in enumerate(loo.split(self.training_set['scaled_input'])):
			mse = []
			for j, regr in enumerate(self.regressors):
				train_input = self.training_set['scaled_input'][train_index]
				train_target = self.training_set['target'][train_index][:, j]
				test_input = self.training_set['scaled_input'][test_index]
				test_target = self.training_set['target'][test_index][:, j]
				regr.fit(train_input, train_target)

				mse.append(np.abs(test_target - regr.predict(test_input)[0]) / test_target)
				# mse = np.square((self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['scaled_input'][test_index])[0]) / self.training_set['target'][test_index][0])

			mse = np.array(mse).flatten()
			all_mse.append(mse)

		return np.average(all_mse, axis=0), all_mse


class PerFeatureGPREmulator(PerFeatureEmulator):

	def __init__(self, training_set):
		super().__init__(training_set, GaussianProcessRegressor)
