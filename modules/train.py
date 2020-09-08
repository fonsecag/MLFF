from run import MainHandler
from .clustererror import ClusterErrorHandler
from .cluster import ClusterHandler
from util import*


class TrainHandler(ClusterErrorHandler):

	curr_model = None
	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)
		self.n_substages = self.n_substages -2 + self.args['n_steps']
		# -1 because no error calc on its own, no plotting

		self.n_stages = self.n_main_stages + self.n_substages
		self.info['training_indices'] = []

	# n_substages = ClusterHandler.n_substages + 1  # needs be dynamic

	def run_command(self):
		ClusterHandler.run_command(self)
		from funcs.models import sgdml_train_default
		# (self, dataset_tuple, n_train, model_path, sgdml_args)
		ext = self.call_para('train_models','model_ext')
		model_path = os.path.join(self.storage_dir,f'model{ext}')
		old_model_path = os.path.join(self.storage_dir,f'old_model')

		## INITIAL MODEL
		self.init_model()
		self.print_stage('Creating initial model')
		self.train_load_model(model_path, self.args['init'])
		if not os.path.exists(model_path):
			shutil.copy( self.args['init'], model_path)
		self.info['training_indices'].append(self.training_indices)

		## ACTUAL ITERATIONS
		n_steps = self.args['n_steps']
		for i in range(n_steps):
			self.print_stage(f'Iterating model ({i+1}/{n_steps})')	

			new_indices = self.find_problematic_indices()

			if os.path.exists(old_model_path):
				if os.path.isdir(old_model_path):
					shutil.rmtree(old_model_path)
				else:
					os.remove(old_model_path)

			# toad: do better
			save_path = f'{old_model_path}_{len(self.training_indices)}{ext}'
			shutil.move(model_path, save_path)

			tr_ind = list(self.training_indices) + list(new_indices)
			self.training_indices = tr_ind

			# # free up memory, sGDML
			# if hasattr(self.curr_model, 'glob_id'):
			# 	glob_id = self.curr_model.glob_id
			# 	if (glob_id is not None) and ("globs" in globals()):
			# 		global globs
			# 		if len(globs)>glob_id:
			# 			globs[glob_id] = None

			self.curr_model = None 
			self.train_load_model(model_path, tr_ind, save_path)
			self.info['training_indices'].append(self.training_indices)

	def find_problematic_indices(self):
		self.calculate_errors(extended = False, sample_wise = True)

		self.fine_indices = self.call_para(
			'fine_clustering', 'fine_indices_func', 
			args=[self, self.cluster_indices, self.cluster_err])

		self.fine_clustering()

		self.calculate_fine_errors()

		new_indices = self.call_para('fine_clustering','indices_func',
			args = [
			self, self.fine_cl_indices, self.sample_err, self.args['step_size']
			])

		return new_indices

	def calculate_fine_errors(self):

		print_subtitle('Calculating error on fine clusters')

		print_ongoing_process('Preparing data')

		#helping variables
		cluster_indices=self.fine_cl_indices
		n_clusters=len(cluster_indices)

		pred_index = self.call_para(
			'predict_error', 'sample_wise_predicts_index')

		pred = self.predicts[pred_index]

		comp_index = self.call_para(
			'predict_error', 'sample_wise_comparison_var_index')

		F_all = self.vars[comp_index]
		print_ongoing_process('Data prepared', True)

		indices = np.concatenate(cluster_indices)
		calculated_indices = np.argwhere(~np.isnan(self.sample_err)).flatten()
		filtered_indices = set(indices) - set(calculated_indices)
		indices_to_calculate = list(filtered_indices)

		n_to_calc = len(indices_to_calculate)
		if n_to_calc > 0:
			print_ongoing_process(f'Predicting {n_to_calc} values')
			F_pred = self.predict_indices(indices_to_calculate, pred_index)
			pred[indices_to_calculate] = F_pred
			F_data = F_all[indices_to_calculate]
			print_ongoing_process(f'{n_to_calc} Values predicted', True)

			err = self.call_para('predict_error', 'sample_wise_error_func',
				args = [self, pred[indices_to_calculate], comp[indices_to_calculate]])

			# save errors 
			self.sample_err[indices_to_calculate] = err 


		print_ongoing_process('Calculating errors')
		self.fine_cluster_err=[self.sample_err[x].mean() 
			for x in cluster_indices]
		print_ongoing_process('Errors on fine clusters calculated', True)

		self.print_fine_error()

	def print_fine_error(self):
		err = self.sample_err 
		ce = self.fine_cluster_err

		summary_table = {}
		summary_table['Min cl. err.'] = f"{np.min(ce):.3f}"
		summary_table['Max cl. err.'] = f"{np.max(ce):.3f}"
		summary_table['Avg cl. err.'] = f"{np.average(ce):.3f}"


		print_table("Fine cluster error summary:",None,None,summary_table, width = 15)

	def fine_clustering(self):
		fine_indices = self.fine_indices 

		from funcs.cluster import cluster_do
		scheme=self.call_para('fine_clustering','clustering_scheme')
		fine_cl_indices = cluster_do(self, scheme, fine_indices)
		self.print_cluster_summary(fine_cl_indices)
		self.fine_cl_indices = fine_cl_indices

	# 'predict_error':{
	# 	'predict_var_index':1,
	# 	'compare_var_index':2,
	# 	'predict_func':'func:sgdml_predict_F',
	# 	'error_func':'func:MSE_sample_wise',
	# }

	def save_command(self):

		super().save_command()
