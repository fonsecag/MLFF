import sys
from .cluster import ClusterHandler
from util import*

class ClusterErrorHandler(ClusterHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)
		self.n_stages = self.n_main_stages + self.n_substages

	n_substages = ClusterHandler.n_substages + 3
	# 3 more than ClusterHandler: load model, calulate error and plot

	def init_model(self):
		# init model
		init = self.args['init']
		self.curr_model = None
		if (type(init)==str) and not (init.isdigit()):
			if os.path.exists(init):
				self.curr_model, self.training_indices = self.call_para(
					'train_models','load_func', args = [self, init]
					)
				if self.curr_model is None:
					self.args['init'] = self.training_indices

			else:
				print_error(f'Could not find model at invalid path {init}')
		else:
			self.args['init'] = int(self.args['init'])

	def run_command(self):
		super().run_command()

		self.init_model()

		self.print_stage('Loading model')
		self.train_load_model(os.path.join(self.storage_dir,'model'),
			self.args['init'], None)

		self.print_stage('Cluster error')	

		self.calculate_errors(extended = True, sample_wise = False)

		self.print_stage('Plotting')
		self.plot_cluster_error()

	def train_load_model(self, model_path, indices, old_model_path = None):

		# indices is actually just the 'init' arg when first loading
		print_subtitle('Training model')
		if self.curr_model is None:

			if (type(indices) == int):
				N = int(indices)
			else:
				N = len(indices)

			print_ongoing_process(f"Training  model ({N} points)")

			self.call_para(
				'train_models','train_func',
				args = [self, indices, model_path, old_model_path]
				)

			self.curr_model, self.training_indices = self.call_para(
				'train_models','load_func', args = [self, model_path]
				)

			print_ongoing_process(f"Trained  model ({N} points)", True)

		else:
			print_ongoing_process('Model file found and loaded', True)



		## PRINT SOME INFO ABOUT MODEL
		if self.get_para('train_models','model_info_func') is not None:
			self.call_para('train_models','model_info_func', 
				args = [self])		

	def predict_indices(self, ind, predict_index):

		input_var = self.call_para(
			'predict_error', 'predicts', predict_index, 'input_var_index')

		var = self.vars[input_var][ind]

		pred = self.call_para(
			'predict_error', 'predicts', predict_index, 'predict_func',
			args = [self, var])
		return pred

	def calculate_errors(self, extended = False, sample_wise = False):
		print_subtitle('Calculating error')

		print_ongoing_process('Preparing data')
		N = np.max(  np.concatenate(self.cluster_indices))
		self.sample_err = None
		self.sample_err = np.empty(N + 1)
		self.sample_err[:] = np.nan
		self.cluster_err = None


		#helping variables
		cluster_indices=self.cluster_indices
		n_clusters=len(cluster_indices)
		print_ongoing_process('Data prepared', True)

		self.predicts = {}
		n_predicts = len( self.get_para('predict_error', 'predicts') )
		n_errors = len( self.get_para('predict_error') )

		# handle extended or not / main
		main_error_index = self.call_para(
			'predict_error', 'main_error_index')

		main_predict_index = self.call_para(
			'predict_error', main_error_index, 'predicts_index')

		if not extended: 
			l_predicts = [main_predict_index]
			l_errors = [main_error_index]

		else:
			l_predicts = np.arange(n_predicts)
			l_errors = np.arange(n_errors)

		print_ongoing_process('Finding sub indices')
		sub_cl_indices = self.call_para('predict_error',
			'error_sub_indices', args = [self, cluster_indices])
		sub_indices = np.concatenate(sub_cl_indices)
		print_ongoing_process('Sub indices found',True)

		for predicts_index in l_predicts:
			print_ongoing_process(
				f'Predicting values for predicts index {predicts_index}')

			pred = self.predict_indices(sub_indices, predicts_index)
			predicts_shape = list(pred.shape)
			predicts_shape[0] = N+1
			predicts_shape = tuple(predicts_shape)

			self.predicts[predicts_index] = np.empty( predicts_shape )
			self.predicts[predicts_index][:] = np.nan			
			self.predicts[predicts_index][sub_indices] = pred 

			print_ongoing_process(
				f'Predicting values for predicts index {predicts_index}', True)


		self.errors = {}
		n_errors = len(self.get_para('predict_error'))

		for error_index in l_errors:

			if self.para['predict_error'].get(error_index, None) is None:
				continue 

			pred_index = self.call_para(
				'predict_error', error_index, 'predicts_index')

			pred = self.predicts[pred_index]

			err_key = self.call_para(
				'predict_error', error_index, 'save_key')

			by_cluster = self.call_para(
				'predict_error', error_index, 'by_cluster')

			comp_index = self.call_para(
				'predict_error', error_index, 'comparison_var_index')

			comp_vars = self.vars[comp_index]
			comp = comp_vars
			err = []
			# print(f'\n{comp_index}  {error_index}')
			if by_cluster or (error_index == main_error_index):
				if not by_cluster:
					print_warning(f'main_error_index should be tied to by_'\
						'cluster, automatically assumed that by_cluster = True')

				for ind in sub_cl_indices:
					# print(len(ind))
					err.append(self.call_para(
						'predict_error', error_index, 'error_func',
						args = [self, pred[ind], comp[ind]] ))
				
			else:
				err = self.call_para('predict_error', error_index, 'error_func',
						args = [self, pred[sub_indices], comp[sub_indices]] )

			if (pred_index == main_predict_index):
				self.cluster_err = err

			self.errors[err_key] = err			


		if sample_wise:

			pred_index = self.call_para(
				'predict_error', 'sample_wise_predicts_index')

			pred = self.predicts[pred_index]

			comp_index = self.call_para(
				'predict_error', 'sample_wise_comparison_var_index')

			comp_vars = self.vars[comp_index]
			comp = comp_vars

			err = self.call_para('predict_error', 'sample_wise_error_func',
				args = [self, pred[sub_indices], comp[sub_indices]])
			self.sample_err[sub_indices] = err 
		
		print_ongoing_process('Errors calculated', True)


		self.info['errors'] = self.errors
		self.print_error(extended = extended, sample_wise = sample_wise)

	def print_error(self, extended = False, sample_wise = False):
		summary_table = {}

		if extended:
			for k, v in self.errors.items():

				if hasattr(v, '__len__'):
					summary_table[f'{k} min'] = f"{np.min(v):.3f}"
					summary_table[f'{k} max'] = f"{np.max(v):.3f}"
					summary_table[f'{k} avg'] = f"{np.average(v):.3f}"

				else:
					summary_table[f'{k}'] = f"{v:.3f}"

		else:
			ce = self.cluster_err
			summary_table['cluster min'] = f"{np.min(ce):.3f}"
			summary_table['cluster max'] = f"{np.max(ce):.3f}"
			summary_table['cluster avg'] = f"{np.average(ce):.3f}"

		if sample_wise:
			err = self.sample_err
			err = err[np.argwhere(~np.isnan(err))]
			summary_table['s-w min'] = f"{np.min(err):.3f}"
			summary_table['s-w max'] = f"{np.max(err):.3f}"
			summary_table['s-w avg'] = f"{np.average(err):.3f}"

		print_table("Cluster error summary:",None,None,summary_table, width = 15)

	def save_command(self):
		super().save_command()
		# info is saved automatically anyways, which contains errors 

	def get_graph_para(self, key, i):

		if self.para['predict_error'].get(i, None) is None:
			return ((key is None) and False) or (None)

		if key is None:
			return self.call_para('predict_error', i, 'graph')

		elif self.para['predict_error'][i]['graph_paras'].get(key, None) is None:
			return self.call_para('error_graph_default', key)

		else:
			return self.call_para('predict_error', i, 'graph_paras', key)

	def plot_cluster_error(self):
		
		import matplotlib.pyplot as plt 

		cluster_indices=self.cluster_indices
		errors = self.errors

		for i in range(len( self.para['predict_error'] )):

			if not self.get_graph_para(None, i):
				continue

			if self.para['predict_error'].get(i, None) is None:
				continue
		
			key = self.call_para('predict_error', i, 'save_key')
			by_cluster = self.call_para('predict_error', i, 'by_cluster')

			if not by_cluster:
				continue

			if key not in errors:
				print_warning(f'Key {key} not found in error file, skipping')
				continue

			err = errors[key]
			if not hasattr(err, '__len__'):
				print_warning(f'Error key {key} is set to by_cluster but error'\
					' is not a list. Skipping.')
				continue

			elif len(cluster_indices) != len(err):
				print_warning(f'Error key {key} is has length {len(err)} but '\
					f' there are {len(cluster_indices)} clusters. Skipping.')		
				continue

			mpl_style = self.get_graph_para('matplotlib_style', i)
			plt.style.use(mpl_style)

			graph_name = f'{key}_plot.pdf'
			graph_path = os.path.join(self.storage_dir, graph_name)
			print_ongoing_process(f'Preparing {graph_name}')

			## ACTUALLY PLOT THE MOFO
			
			# SORTING
			if self.get_graph_para('order_by_error', i):
				ind_sorted = np.argsort(err)
			else:
				ind_sorted = np.arange(len(err))

			err = np.array(err)[ind_sorted]
			cl_ind = np.array(cluster_indices, dtype = 'object')[ind_sorted]

			# HELPING VARS
			n_clusters = len(cl_ind)
			min_err, max_err,avg_err = np.min(err), np.max(err), np.average(err)
			med_err, mm_diff = (max_err+min_err)/2, max_err-min_err			

			# X AXIS
			x = np.arange(n_clusters) + 1
			if self.get_graph_para('reverse_order', i):
				x = np.flip(x)


			# POPULATION / TRUE AVG
			pop = np.array([len(y) for y in cl_ind])
			pop_tot = np.sum(pop)
			if self.get_graph_para('horizontal_cluster_average', i):
				real_avg = avg
			else:
				if self.get_graph_para('real_average_key',i) is not None:
					try:
						key = self.get_graph_para('real_average_key',i)
						err_avg = self.errors[key]
						real_avg = err_avg
					except:
						key = self.get_graph_para('real_average_key',i)
						print_warning(f'Error key {key} not found, skipped')
						real_avg = np.sum(np.array(pop) * np.array(err)) / pop_tot
				else:
					real_avg = np.sum(np.array(pop) * np.array(err)) / pop_tot

			# RESCALE POP FOR GRAPH 
			pop = pop * med_err / np.max(pop)			


			# HANDLE COLOR
			bar_color = self.get_graph_para('bar_color', i)
			color_steps = self.get_graph_para("bar_color_gradient", i)
			# only used for gradients
			
			if len(color_steps)<2:
				print_warning(f"Parameter color_steps should be >1,"\
					f" is {len(color_steps)}.")


			if bar_color=='cluster_gradient':
				bar_color = color_interp(color_steps, np.arange(n_clusters))

			elif bar_colors == "error_gradient":
				bar_color= color_interp(color_steps, err)


			# CREATE FIGURE
			figsize = self.get_graph_para('figsize', i)
			fig = plt.figure(figsize = figsize)
			ax = fig.add_subplot(111)


			ax.bar(x, err, color = bar_color, 
				width = self.get_graph_para('bar_width', i))

			ax.set_xlim(0 + n_clusters*0.01, n_clusters*1.01)
		
			#create population line IF asked for
			if self.get_graph_para('include_population', i):

				fontsize = self.get_graph_para('population_fontsize', i)
				color = self.get_graph_para('population_color', i)
				linewidth = self.get_graph_para('population_linewidth', i)

				ax.step(x, pop, c = color, where = "mid", 
					linewidth = linewidth, 
					alpha = self.get_graph_para('population_alpha', i))

				y = np.max(pop)
				ax.text(0, y, self.get_graph_para('population_text', i),
					fontsize = fontsize,
					color = color,
					horizontalalignment = "left",
					verticalalignment = "bottom")

			# HOR AVG LINE
			if self.get_graph_para('horizontal_line', i):

				lw = self.get_graph_para('horizontal_line_linewidth', i)
				fontsize = self.get_graph_para('horizontal_label_fontsize', i) 
				color = self.get_graph_para('horizontal_line_color', i)

				ax.axhline(real_avg, linewidth = lw, color = color)

				if self.get_graph_para('horizontal_label_text' ,i) is not None:
					ax.text(.8, real_avg+(max_err-min_err) * 0.03,
						self.get_graph_para('horizontal_label_text', i),
						fontsize = fontsize,
						color = color,
						horizontalalignment = "left")

			xlabel = self.get_graph_para('x_axis_label', i)
			ylabel = self.get_graph_para('y_axis_label', i)
			fontsize = self.get_graph_para('axis_label_fontsize', i)
			ax.set_xlabel(xlabel, fontsize = fontsize)
			ax.set_ylabel(ylabel, fontsize = fontsize)

			if self.get_graph_para('label_cluster_index', i):
				fs = self.get_graph_para('cluster_index_label_fontsize', i)
				e_max = np.max(err)
				for j in range(len(err)):
					e = err[j]
					plt.text(j + 1, e + 0.005*e_max, f'{ind_sorted[j]}',
						fontsize = fs,
						ha = 'center',
						va = 'bottom')

			print_ongoing_process(f'Preparing {graph_name}', True)
			print_ongoing_process(f'Saved graph at {graph_path}',True)
			transparent = self.get_graph_para('transparent_background', i)
			DPI = self.get_graph_para('DPI', i)

			ax.set_title(key, 
				fontsize = self.get_graph_para('title_fontsize', i))
			fig.savefig(graph_path, DPI = DPI)