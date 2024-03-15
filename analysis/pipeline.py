import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from analysis.map import Map
from analysis.persistence_diagram import PersistenceDiagram
from analysis.persistence_diagram import BettiNumbersGridVarianceMap, PixelDistinguishingPowerMap


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
	from tqdm.notebook import tqdm
else:
	from tqdm import tqdm


class Pipeline:

	def __init__(self, 
			  maps_dir='maps', 
			  force_recalculate=False, 
			  filter_region=None, 
			  filter_cosmology=None, 
			  filter_los=None, 
			  do_remember_maps=True, 
			  save_plots=False, 
			  bng_resolution=100,
			  three_sigma_mask=False
		):
		self.maps_dir = maps_dir
		self.recalculate = force_recalculate
		self.filter_region = filter_region if filter_region is not None else '*'
		self.filter_cosmology = filter_cosmology
		self.filter_los = filter_los if filter_los is not None else '*'
		self.do_remember_maps = do_remember_maps
		self.save_plots = save_plots
		self.bng_resolution = bng_resolution
		self.three_sigma_mask = three_sigma_mask

	def run_pipeline(self):
		self.find_max_min_values_maps()
		self.read_maps()
		self.calculate_variance()

	def _get_glob_str_dir(self):
		return f'{self.maps_dir}/*SN0*' if self.filter_cosmology is None else f'{self.maps_dir}/*SN0*{self.filter_cosmology}'

	def _get_glob_str_file(self, dir):
		# file_f = '*SN0*.npy' if self.filter_region is None else f'*SN0*R{self.filter_region}.S*.npy'
		# dir_f = f'{dir}' if self.filter_cosmology is None else f'{dir}'
		# return f'{dir}/*SN0*.npy' if self.filter_region is None else f'{dir}/*SN0*R{self.filter_region}.S*.npy'
	
		return dir + f'/SN0*LOS{self.filter_los}R{self.filter_region}.S*.npy'

	def find_max_min_values_maps(self, save_all_values=False, save_maps=False):		
		print('Determining max and min values in maps...')

		if not self.recalculate and os.path.exists(os.path.join(self.maps_dir, 'extreme_values.json')):
			print('Found file with saved values, reading...')
			with open(os.path.join(self.maps_dir, 'extreme_values.json')) as file:
				data_range_read = json.loads(file.readline())

				# JSON format does not allow for int as key, so we change from str keys to int keys
				self.data_range = {dim: data_range_read[str(dim)] for dim in [0, 1]}
				return self.data_range
		
		vals = {
			'min': np.inf,
			'max': -np.inf
		}

		self.all_values = []
		self.all_values_cosm = {}
		self.all_maps = []
		
		for dir in tqdm(glob.glob(self._get_glob_str_dir())):
			if os.path.isdir(dir):
				for i, map_path in enumerate(glob.glob(self._get_glob_str_file(dir))):
					if 'LOS0' in map_path:# or 'LOS10' in map_path or 'LOS46' in map_path:
						continue

					map = Map(map_path, three_sigma_mask=self.three_sigma_mask)

					curr_min = np.min(map.map[np.isfinite(map.map)])
					curr_max = np.max(map.map[np.isfinite(map.map)])

					# Diagnostic check
					if curr_min < -.1:
						print('min check', map_path)
					if curr_max > .1:
						print('max check', map_path)
					
					if curr_min < vals['min']:
						vals['min'] = curr_min
					if curr_max > vals['max']:
						vals['max'] = curr_max

					if save_all_values:
						self.all_values += map.map[np.isfinite(map.map)].flatten().tolist()
						# self.all_values_cosm[map.cosmology_id] = map.map[np.isfinite(map.map)].flatten().tolist()

					if save_maps:
						self.all_maps.append(map)
		
		print('min=', vals['min'])
		print('max=', vals['max'])

		self.data_range = {
			dim : [vals['min'], vals['max']] for dim in [0, 1]
		}

		with open(os.path.join(self.maps_dir, 'extreme_values.json'), 'w') as file:
			file.write(json.dumps(self.data_range))

		return self.data_range

	def all_values_histogram(self, ax=None):		
		if ax is None:
			fig, ax = plt.subplots()
		ax.hist(self.all_values, bins=500)
		# ax.semilogy()

		# fig, ax = plt.subplots()
		# hists = []
		# bine = np.arange(-.05, .05, .001)
		# binc = (bine + .001)[:-1]
		# for csm, vals in self.all_values_cosm.items():
		# 	h, _ = np.histogram(vals, bins=bine)
		# 	hists.append(h)
		# 	ax.scatter(binc, h, color='grey', s=2, alpha=.3)
		
		# ax.scatter(binc, np.average(hists, axis=0), color='red', s=2)
		
		# ax.legend()

	def read_maps(self):

		print(self.data_range)

		# SLICS determines the sample variance, will be a list of persistence diagrams for each line of sight
		self.slics_pds = []
		# cosmoSLICS is different cosmologies, will be a list of persistence diagrams for each cosmology
		self.cosmoslics_pds = []
		# cosmoslics_uniq_pds = []

		self.slics_maps = []
		# cosmoslics_maps = []

		do_delete_maps = not self.do_remember_maps

		print('Analyzing maps...')
		for dir in tqdm(glob.glob(self._get_glob_str_dir())):
			if os.path.isdir(dir):
				cosm = dir.split('_')[-1]

				cosmoslics = 'Cosmo' in cosm

				curr_cosm_maps = []

				glob_str = self._get_glob_str_file(dir)

				for i, map_path in enumerate(tqdm(glob.glob(glob_str), leave=False)):
					if 'LOS0' in map_path:# or 'LOS10' in map_path or 'LOS46' in map_path:
						continue
					map = Map(map_path, three_sigma_mask=self.three_sigma_mask)
					map.get_persistence()
					curr_cosm_maps.append(map)


					# SLICS must be saved at LOS level
					if not cosmoslics:
						perdi = PersistenceDiagram([map], do_delete_maps=do_delete_maps)
						perdi.generate_betti_numbers_grids(resolution=self.bng_resolution, data_ranges_dim=self.data_range, regenerate=self.recalculate, save_plots=self.save_plots)
						self.slics_pds.append(perdi)
						self.slics_maps.append(map)
					else:
						pass
						# cosmoslics_uniq_pds.append(perdi)
						# cosmoslics_maps.append(map)

				if len(curr_cosm_maps) > 0 and cosmoslics:
					perdi = PersistenceDiagram(curr_cosm_maps, do_delete_maps=do_delete_maps)
					# pd.generate_heatmaps(resolution=100, gaussian_kernel_size_in_sigma=3)
					# pd.add_average_lines()
					perdi.generate_betti_numbers_grids(resolution=self.bng_resolution, data_ranges_dim=self.data_range, regenerate=self.recalculate, save_plots=self.save_plots)

					if self.save_plots:
						perdi.plot()

					# cosmoSLICS must be saved at cosmology level
					if cosmoslics:
						self.cosmoslics_pds.append(perdi)

		return self.slics_pds, self.cosmoslics_pds

	def calculate_variance(self):
		print('Calculating SLICS/cosmoSLICS variance maps...')
		slics_bngs = {
			dim: [spd.betti_numbers_grids[dim] for spd in self.slics_pds] for dim in [0, 1]
		}
		cosmoslics_bngs = {
			dim: [cpd.betti_numbers_grids[dim] for cpd in self.cosmoslics_pds] for dim in [0, 1]
		}

		dim = 0
		slics_bngvm_0 = BettiNumbersGridVarianceMap(slics_bngs[dim], birth_range=self.data_range[dim], death_range=self.data_range[dim], dimension=dim)
		
		dim = 1
		slics_bngvm_1 = BettiNumbersGridVarianceMap(slics_bngs[dim], birth_range=self.data_range[dim], death_range=self.data_range[dim], dimension=dim)

		dim = 0
		cosmoslics_bngvm_0 = BettiNumbersGridVarianceMap(cosmoslics_bngs[dim], birth_range=self.data_range[dim], death_range=self.data_range[dim], dimension=dim)

		dim = 1
		cosmoslics_bngvm_1 = BettiNumbersGridVarianceMap(cosmoslics_bngs[dim], birth_range=self.data_range[dim], death_range=self.data_range[dim], dimension=dim)

		if self.save_plots:
			slics_bngvm_0.save_figure(os.path.join('plots', 'slics'), title='SLICS variance, dim=0')
			slics_bngvm_1.save_figure(os.path.join('plots', 'slics'), title='SLICS variance, dim=1')
			cosmoslics_bngvm_0.save_figure(os.path.join('plots', 'cosmoslics'), title='cosmoSLICS variance, dim=0')
			cosmoslics_bngvm_1.save_figure(os.path.join('plots', 'cosmoslics'), title='cosmoSLICS variance, dim=1')

		slics_pd = PersistenceDiagram(self.slics_maps)
		slics_pd.generate_betti_numbers_grids(data_ranges_dim=self.data_range, regenerate=self.recalculate, resolution=self.bng_resolution)

		self.dist_powers = []

		for dim in [0, 1]:
			slics_var_map = BettiNumbersGridVarianceMap(slics_bngs[dim], birth_range=self.data_range[dim], death_range=self.data_range[dim], dimension=dim)
			slics_var_map.save(os.path.join('products', 'bng_variance', 'slics'))

			dist_power = PixelDistinguishingPowerMap([cpd.betti_numbers_grids[dim] for cpd in self.cosmoslics_pds], slics_pd.betti_numbers_grids[dim], slics_var_map, dimension=dim)
			dist_power.save(os.path.join('products', 'pixel_distinguishing_power'))
			
			if self.save_plots:
				slics_var_map.save_figure(os.path.join('plots', 'bng_variance', 'slics'))
				dist_power.save_figure(os.path.join('plots', 'pixel_distinguishing_power'))

			self.dist_powers.append(dist_power)

		del slics_pd
		del self.slics_maps

		return self.dist_powers