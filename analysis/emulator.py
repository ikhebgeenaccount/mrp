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
		return self.regressor.predict(X)
	
	def validate(self):
		loo = LeaveOneOut()

		curr_mse = np.zeros(shape=self.training_set['scaled_target'].shape[1])

		for i, (train_index, test_index) in enumerate(loo.split(self.training_set['scaled_input'])):
			self.regressor.fit(self.training_set['scaled_input'][train_index], self.training_set['target'][train_index])
			
			curr_mse += np.square(self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['scaled_input'][test_index])[0])
		
		curr_mse = curr_mse / curr_mse.shape[0]

		return curr_mse


class GPREmulator(Emulator):

	def __init__(self, training_set) -> None:
		super().__init__(training_set)
		self.regressor = GaussianProcessRegressor()


class MLPREmulator(Emulator):

	def __init__(self, training_set) -> None:
		super().__init__(training_set)
		self.regressor = MLPRegressor()

	def hyperparameter_optimization(self, param_grid):
		# TODO
		rcv = RandomizedSearchCV()
		pass