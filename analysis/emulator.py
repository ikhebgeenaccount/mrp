import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

class Emulator:

	def __init__(self, training_set):
		self.training_set = training_set
		self.regressor = None

		self.standard_scaler = StandardScaler()
		# Ensure that training_set contains numpy.ndarrays
		self.training_set['input'] = np.array(self.training_set['input'])
		self.training_set['scaled_input'] = self.standard_scaler.fit_transform(self.training_set['input'])
		self.training_set['target'] = np.array(self.training_set['target'])
		# self.training_set['scaled_target'] = self.standard_scaler.fit_transform(self.training_set['target'])

	def fit(self):
		self.regressor.fit(self.training_set['input'], self.training_set['target'])
	
	def predict(self, X):
		# scale X through self.standard_scaler, not fitting again
		return self.regressor.predict(self.standard_scaler.transform(X))
	
	def validate(self):
		loo = LeaveOneOut()

		all_mse = []

		for i, (train_index, test_index) in enumerate(loo.split(self.training_set['scaled_input'])):
			self.regressor.fit(self.training_set['scaled_input'][train_index], self.training_set['target'][train_index])

			mse = np.abs(self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['scaled_input'][test_index])[0]) / self.training_set['target'][test_index][0]
			# mse = np.square((self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['scaled_input'][test_index])[0]) / self.training_set['target'][test_index][0])

			all_mse.append(mse)

		return np.average(all_mse, axis=0), all_mse

	def hyperparameter_optimization(self, param_grid):
		# TODO
		rcv = RandomizedSearchCV()
		pass


class GPREmulator(Emulator):

	def __init__(self, training_set) -> None:
		super().__init__(training_set)
		self.regressor = GaussianProcessRegressor()


class MLPREmulator(Emulator):

	def __init__(self, training_set) -> None:
		super().__init__(training_set)
		self.regressor = MLPRegressor()


class PerFeatureEmulator(Emulator):
	"""
	Instead of having one GPR for the whole data vector, we have one for each entry."""

	def __init__(self, training_set, regressor_type):
		super().__init__(training_set)

		# Create an emulator for each entry in the target data vector
		self.regressors = [regressor_type() for _ in training_set['target'][0]]
	
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
