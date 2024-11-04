import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

from analysis.data_compression.compressor import Compressor
from analysis import cosmologies

class Emulator:

	def __init__(self, regressor_type, training_set=None, compressor: Compressor=None, plots_dir='plots', **regressor_args):
		if training_set is not None:
			self.compressor = None
			self.training_set = training_set
		elif compressor is not None:
			self.compressor = compressor
			self.training_set = compressor.cosmoslics_training_set

		self.regressor_type = regressor_type
		self.regressor_args = regressor_args
		self.regressor = regressor_type(**regressor_args)

		# Ensure that training_set contains numpy.ndarrays
		self.training_set['input'] = np.array(self.training_set['input'])
		self.training_set['target'] = np.array(self.training_set['target'])

		self.data_vector_length = self.training_set['target'].shape[1]

		self.plots_dir = plots_dir

	def fit(self):
		self.regressor.fit(self.training_set['input'], self.training_set['target'])
	
	def predict(self, X):
		return self.regressor.predict(X)
	
	def validate(self, make_plot=False):
		loo = LeaveOneOut()

		all_mse = []

		for i, (train_index, test_index) in enumerate(loo.split(self.training_set['input'])):
			temp_regr = self.regressor_type(**self.regressor_args)
			temp_regr.fit(self.training_set['input'][train_index], self.training_set['target'][train_index])

			# Not np.abs to also get negative error
			mse = (self.training_set['target'][test_index][0] - temp_regr.predict(self.training_set['input'][test_index])[0])
			# mse = np.square((self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['input'][test_index])[0]) / self.training_set['target'][test_index][0])

			all_mse.append(mse)

		avg_mse = np.nanmean(all_mse, axis=0)

		if make_plot:
			self.create_loocv_plot(avg_mse, all_mse)

		return avg_mse, all_mse

	def hyperparameter_optimization(self, param_grid):
		# TODO
		rcv = RandomizedSearchCV()
		pass

	def _save_plot(self, fig, save_name):
		fig.tight_layout()
		fig.savefig(f'{self.plots_dir}/{type(self.compressor).__name__}_{type(self.regressor).__name__}_{save_name}.pdf')
		plt.close(fig)

	def create_loocv_plot(self, avg_mse, all_mse, plot_cov=True, logy=False):
		fig, axs = self.compressor.plot_data_vectors(include_slics=False, include_cosmoslics=False, save=False)

		nrows = fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().nrows

		for row_id in range(nrows):
			if nrows == 1:
				ax = axs
			else:
				ax = axs[row_id]

			vector_slice = np.s_[row_id * 50:min(self.data_vector_length, (row_id +1) * 50)]
			x = np.arange(vector_slice.start, vector_slice.stop)
		
			# ax.set_title(f'{self.training_set["name"]} after LOOCV')
			ax.set_ylabel('Error in $\sigma_\mathrm{SLICS}$')
			
			ax.plot(x, avg_mse[vector_slice] / self.compressor.avg_slics_data_vector_err[vector_slice], label='Average error', color='red', linewidth=3)

			ax.plot(x, np.array(all_mse / self.compressor.avg_slics_data_vector_err).T[vector_slice], color='blue', alpha=.2, linewidth=1)

			# Dummy LOO realisation for legend
			ax.plot(np.nan, color='blue', alpha=.2, linewidth=1, label='Single LOO realisation')
			
			# Add covariance matrix shaded area
			if self.compressor is not None and plot_cov:
				cov = np.ones(self.data_vector_length)
				ax.fill_between(x=x, y1=-cov[vector_slice], y2=cov[vector_slice], color='grey', alpha=.4, label='$1\sigma$ covariance')

			if row_id == 0:
				ax.legend()
	
		ax.set_xlabel('Data vector entry')

		self._save_plot(fig, 'loocv')
		return fig, ax
	
	def plot_predictions_over_parameters(self, preds_count=10, colormap='viridis', save=True):
		for i in range(4):
			self.plot_predictions_over_input_index(i, preds_count, colormap, save)

	def plot_predictions_over_input_index(self, index, index_preds=10, colormap='viridis', save=True):
		index_ranges = [
			[0.1, 0.55], 
			[0.6, 0.9],
			[0.6, 0.9],
			[-2.0, 0.5]
		]

		index_names = ['Omega_M', 'S_8', 'h', 'w_0']
		lbls = ['$\Omega_M$', '$S_8$', '$h$', '$w_0$']

		index_range = index_ranges[index]

		fig, axs = self.compressor.plot_data_vectors(include_slics=True, include_cosmoslics=False, save=False, true_value=False)

		param_values = np.linspace(*index_range, index_preds)

		cosm_params = np.broadcast_to(self.compressor.slics_training_set['input'][0], (index_preds, 4)).copy()
		# 2nd entry is S_8
		cosm_params[:, index] = param_values

		predictions = self.predict(cosm_params)

		# Add the predictions to the plot
		cmap = mpl.colormaps[colormap]
		norm = mpl.colors.Normalize(vmin=index_range[0], vmax=index_range[1])
		# ScalarMappable for colorbar
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])

		nrows = fig.axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec().nrows

		
		cbar = fig.colorbar(sm, ax=axs if nrows == 1 else axs[-1])
		cbar.set_label(lbls[index])
		cbar.ax.axhline(self.compressor.slics_training_set['input'][0][index], color='black', linestyle='dotted')

		for row_id in range(nrows):
			if nrows == 1:
				ax = axs
			else:
				ax = axs[row_id]

			vector_slice = np.s_[row_id * 50:min(self.data_vector_length, (row_id +1) * 50)]
			x = np.arange(vector_slice.start, vector_slice.stop)

			for i, pred in enumerate(predictions):
				ax.plot(x, ((pred - self.compressor.avg_slics_data_vector) / self.compressor.avg_slics_data_vector_err)[vector_slice], c=cmap(norm(param_values[i])))

		if save:
			self._save_plot(fig, f'predictionss_over_{index_names[index]}')

	def plot_predictions_over_s8(self, s8_count=10, colormap='viridis', save=True):
		self.plot_predictions_over_input_index(1, s8_count, colormap, save)
	
	def plot_data_vector_over_param_space(self, base_cosmology_id):		
		fig, axs = plt.subplots(nrows=self.data_vector_length, ncols=4, figsize=(30, 4 * self.data_vector_length), sharex='col', sharey='row')
		fig.suptitle(f'Using cosmology {base_cosmology_id}')

		if self.training_set['name'] == 'number_of_features':
			axs[0][0].set_ylabel('Connected components count')
			axs[1][0].set_ylabel('Holes count')

		truths = cosmologies.get_cosmological_parameters(base_cosmology_id).values[0][1:]

		for i, param in enumerate(['Omega_m', 'S_8', 'h', 'w_0']):
			axs[0][i].set_title(f'{param}')
			param_index = i
			param_values = np.linspace(
				np.min(self.compressor.cosmoslics_training_set['input'][:, param_index]), 
				np.max(self.compressor.cosmoslics_training_set['input'][:, param_index]), 
				1000
			)

			# Take base_cosmology_id's cosmological parameters as base cosmology
			cosm_params_base = cosmologies.get_cosmological_parameters(base_cosmology_id)

			# Generate new cosmological parameter sets
			cosm_params_sets = []
			for val in param_values:
				temp_set = cosm_params_base[['Omega_m', 'S_8', 'h', 'w_0']].values.copy()
				temp_set[0][param_index] = val
				cosm_params_sets.append(temp_set)

			cosm_params_sets = np.concatenate(cosm_params_sets)

			# Predict number of features for each cosm param set
			prediction = self.predict(cosm_params_sets)

			for entry in range(self.data_vector_length):
				axs[entry][i].plot(param_values, prediction[:, entry])
				axs[entry][i].axvline(x=truths[i], linestyle='--', color='black')
				# axs[entry][i].axhline(y=truths[i], linestyle='--', color='black')


class GPREmulator(Emulator):

	def __init__(self, training_set=None, compressor=None, **regressor_args) -> None:
		super().__init__(GaussianProcessRegressor, training_set=training_set, compressor=compressor, normalize_y=True, **regressor_args)


class MLPREmulator(Emulator):

	def __init__(self, training_set) -> None:
		super().__init__(MLPRegressor, training_set)


class PerFeatureEmulator(Emulator):
	"""
	Instead of having one GPR for the whole data vector, we have one for each entry."""

	def __init__(self, regressor_type, compressor, **regressor_kwargs):
		super().__init__(regressor_type, compressor=compressor, **regressor_kwargs)

		# Create an emulator for each entry in the target data vector
		self.regressors = [regressor_type(**self.regressor_args) for _ in self.training_set['target'][0]]
	
	def fit(self):
		pass

	def predict(self, X):
		# TODO: fix for multiple predictions in X at once (multiple input vectors in X)
		return np.array([[regressor.fit(X).flatten()[0] for regressor in self.regressors]])

	def validate(self, make_plot=False):
		loo = LeaveOneOut()

		all_mse = []

		for i, (train_index, test_index) in enumerate(loo.split(self.training_set['input'])):
			mse = []
			for j, regr in enumerate(self.regressors):
				train_input = self.training_set['input'][train_index]
				train_target = self.training_set['target'][train_index][:, j]
				test_input = self.training_set['input'][test_index]
				test_target = self.training_set['target'][test_index][:, j]
				regr.fit(train_input, train_target)

				mse.append((test_target - regr.predict(test_input)[0]) / test_target)
				# mse = np.square((self.training_set['target'][test_index][0] - self.regressor.predict(self.training_set['input'][test_index])[0]) / self.training_set['target'][test_index][0])

			mse = np.array(mse).flatten()
			all_mse.append(mse)

		avg_mse = np.average(all_mse, axis=0)

		if make_plot:
			self.create_loocv_plot(avg_mse, all_mse)

		return avg_mse, all_mse


class PerFeatureGPREmulator(PerFeatureEmulator):

	def __init__(self, compressor: Compressor, **regressor_kwargs):
		super().__init__(GaussianProcessRegressor, compressor=compressor, normalize_y=True, **regressor_kwargs)
