

parameters={	
	'n_cores':-2, 
	# The number of cores to use for (some) functions
	# Most of the libraries/functions used in this work handle this on their own
	# This parameter only affects handmade functions where multiprocessing is
	# handled manually.
	# Negative numbers means number of total cores minus that number


	'remove_temp_files':True,
	# A temp folder is created during the execution of many commands, containing
	# temporary files and folders. This is usally deleted after the program is
	# done. Setting this option to False avoids the deletion of the temp folder

	'load_dataset':{
		'post_processing':None, 
		'post_processing_args':[], 

		'schnet_preprocessed_npz':False,

		'var_funcs':{
			0:'func:r_to_dist',
			1:'func:extract_R_concat',
			2:'func:extract_F_concat',
			3:'func:extract_E',
		}, #end of 'var_funcs'
	},


	'clusters':{
		'save_npy' : True,
		'init_scheme':[0, 1],
		#indices below define the clustering scheme
		0:{
			'type':'func:agglomerative_clustering', 
			'n_clusters':10,
			'initial_number':20000,

			'custom': False, ## distance_matrix_func and linkage only active if True
			'distance_matrix_function':'func:distance_matrix_euclidean',
			'linkage':'complete',

			'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
			'var_index':0,
			},
		1:{
			'type':'func:kmeans_clustering',
			'n_clusters':5,
			'var_index':3,
		},
		2:{
			'type':'func:agglomerative_clustering', 
			'n_clusters':100,
			'initial_number':10000,
			'custom':False,
			'distance_matrix_function':'func:distance_matrix_euclidean',
			'linkage':'complete',
			'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
			
			'var_index':0,
			},
	}, #end of 'clusters'

	'classification':{
		'var_index':0,
		'scaler_func':'func:standard_scaler',  #can be None
		# 'scaler_func':None,

		## RANDOM FOREST -- 0.81 / 1.00
		'class_func':'func:random_forest_classifier', 
		'n_estimators':100, # 100-200
		'max_depth':20, # 20
		'min_samples_split':2, # 2 
		'criterion':'entropy', # dm
		'min_impurity_decrease':0, # 0
		'ccp_alpha':0, # 0 


		## EXTREME FOREST -- 0.81 / 1.00
		# 'class_func':'func:extreme_forest_classifier', 
		# 'n_estimators':100, # 100-200
		# 'max_depth':20, # 20
		# 'min_samples_split':2, # 2 
		# 'criterion':'entropy', # dm

		## SVM -- 0.59 / 0.60
		# 'class_func':'func:svm_svc_classifer',
		# 'reg':0.01,

		## Gaussian Process Clf -- 0.71 / 0.85 (also takes years)
		# 'class_func':'func:gaussian_process_classifier', 

		# NN Clf -- 0.83 / 0.99 (relu, adam, constant)
		# 'class_func':'func:neural_network_classifier',
		# 'hidden_layers':(250, 250),
		# 'alpha':.01,
		# 'learning_rate':'constant',
		# 'solver':'adam',
		# 'activation':'relu',

		## AdaBoost Clf -- 0.61 / 0.61
		# 'class_func':'func:AdaBoost_classifier',
		# 'n_estimators':200,
		# 'learning_rate':1,

		'n_points':.7, 
		'perform_test':True,
		'test_func':'func:dotscore_classifier_test',
	},

	'storage':{
		'storage_dir':'saves',
		'save_para_user':True,
		'save_para_default':True,
		'save_original_call':True,
		'dir_name':'func:get_dataset_basename',
		'dir_name_args':['func:spl_return_self'],
	},

	'sub_datasets':{
		'n_subdata':'func:get_splitter_n_clusters',  #args are just the splitter
		'indices_func':'func:get_splitter_cluster_indices_i', #function to be called every step of the loop
											#returns indices of the initial dataset to be used as a subset
											#args are (db,i,*args) where 'i' is the iteration variable
		'save_func':'func:save_subdata_sgdml_i',
	},

	'train_models':{
		# 
		'train_func':'func:sgdml_train_default',
		'model_ext':'.npz',
		'train_func_args':['para:train_models,sgdml_train_args'],
		'suppress_sgdml_prints':True,
		'load_func':'func:load_sgdml_model', # needs to also load trianing set
		'model_info_func':'func:sgdml_model_info',
		'sgdml_train_args':{
			'n_train':50, # will be handled automatically where needed
						   # for example for improved learning
			'n_test':100,
			'n_valid':100,
			'overwrite':True,
			'command':'all',
			'sigs':None,
			'gdml':False,
			'use_E':False,
			'use_E_cstr':False,
			# 'max_processes':-1, # set manually in the function
			'use_torch':False,
			'solver':'analytic',
			'use_cprsn':False,
		},
	},


	'predict_error':{
		'error_sub_indices':'func:return_second_argument', #twenty_each
		'main_error_index':0, 
		# the error that is used for those commands that only need one type 
		# of error (like 'train') rather than an entire analysis
		# error will be saved into self.cluster_err (by_cluster must be true!)
		# only main error is calculated if extended = False in calculate_error

		'sample_wise_error_func':'func:MSE_sample_wise',
		'sample_wise_predicts_index':0,
		'sample_wise_comparison_var_index':2,
		# sample-wise only relevant if sample_wise = True in calculate_error

		'predicts':{
			0:{
				'predict_func':'func:sgdml_predict_F',
				'batch_size':100,
				'input_var_index':1,
			},

		}, # end of predicts


		#RMSE
		0:{
			'predicts_index':0,
			'comparison_var_index':2,			


			'by_cluster':True,
			'error_func':'func:root_mean_squared_error',
			'file_name':'RMSE_graph',
			'save_key':'RMSE_c', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':True,
			'graph_paras':{
				'y_axis_label':r'Force prediction error $(kcal/mol\,\AA)$',
				'real_average_key':'RMSE_o'
				#replace default paras here
			},
		},

		#overall RMSE
		1:{
			'predicts_index':0,
			'comparison_var_index':2,			

			'error_func':'func:root_mean_squared_error',
			'save_key':'RMSE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		},

		#MAE
		2:{
			'predicts_index':0,
			'comparison_var_index':2,			

			'by_cluster':True,
			'error_func':'func:mean_absolute_error',
			'file_name':'MAE_graph',
			'save_key':'MAE_c', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':True,
			'graph_paras':{
				'y_axis_label':r'Force prediction error $(kcal/mol\,\AA)$',
				'real_average_key':'MAE_o'
				#replace default paras here
			},
		},

		#overall MAE
		3:{
			'predicts_index':0,
			'comparison_var_index':2,			

			'error_func':'func:mean_absolute_error',
			'save_key':'MAE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		}, 

		#MSE
		4:{
			'predicts_index':0,
			'comparison_var_index':2,			

			'by_cluster':True,
			'error_func':'func:mean_squared_error',
			'file_name':'MSE_graph',
			'save_key':'MSE_c', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':True,
			'graph_paras':{
				'y_axis_label':r'Force prediction error $(kcal/mol\,\AA)^2$',
				'real_average_key':'MSE_o'
			},
		},

		#overall MSE
		5:{
			'predicts_index':0,
			'comparison_var_index':2,			

			'error_func':'func:mean_squared_error',
			'save_key':'MSE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		},

		# # RMSE atom-wise
		# 6:{
		# 	'predicts_index':0,
		# 	'comparison_var_index':2,			

		# 	'error_func':'func:root_mean_squared_error_atom_wise',
		# 	'save_key':'RMSE_atomwise', #will be saved in save.p under the given key (in 'errors' sub dict)
		# 	'graph':False,
		# 	'graph_paras':{
		# 	},
		# },		

		# # RMSE atom-wise-bins
		# 7:{
		# 	'predicts_index':0,
		# 	'comparison_var_index':2,			

		# 	'error_func':'func:root_mean_squared_error_atom_wise_bins',
		# 	'save_key':'RMSE_atomwise_bins', #will be saved in save.p under the given key (in 'errors' sub dict)
		# 	'graph':False,
		# 	'graph_paras':{
		# 	},
		# },		
	},

	'error_graph_default':{
		'matplotlib_style':'seaborn',

		'reverse_order':False,  #if False, orders from lowest to highest (left to right), reversed otherwise
		'order_by_error':True,
		'label_cluster_index':True,
		'cluster_index_label_fontsize':6,

		# AXES
		'x_axis_label':'Cluster index',
		'y_axis_label':'Prediction error',
		'axis_label_fontsize':12,

		# HOR LINE
		'horizontal_cluster_average':False,
		'real_average_key':None,
		'horizontal_line':True,
		'horizontal_line_linewidth':1,
		'horizontal_label_fontsize':10,
		'horizontal_line_color':'black',
		'horizontal_label_text':'Mean',

		# POPULATION
		'include_population':True,
		'population_fontsize':10,
		'population_color':'blue',
		'population_linewidth':1,
		'population_alpha':1,
		'population_text':'',

		# BAR VISUALS
		'bar_color':'cluster_gradient', # error_gradient
		'bar_color_gradient':[(.2,.8,.2),(.8,.8,.2),(.8,.2,.2)],
		'bar_width':.93,


		# MISC
		'title_fontsize':18,
		'figsize':(9, 6),
		'transparent_background':False,
		'DPI':300,

		# 'x_axis_label':"Cluster index",
		# 'y_axis_label':r'Force prediction error ($?$)',
		# 'axis_label_fontsize':10,
		# 'axis_tick_size':10,
		# 'axis_linewidth':1.3,

		# 'bar_width':.95,
		# 'bar_color':'error_gradient', #cluster_gradient
		# 'bar_color_gradient':[(.2,.8,.2),(.8,.8,.2),(.8,.2,.2)],  #,(.5,.2,.1)

		# 'horizontal_line':True, #if True, includes a horizontal line to show the average error 
		# 'horizontal_line_linewidth':1,
		# 'horizontal_line_color':(.2,.2,.2), 
		# 'horizontal_label_fontsize':9,
		# 'horizontal_label_text':'overall error',


		# 'fig_size':(5,5), #fig size
		# 'transparent_background':False ,

		# 'include_population':True,  #indicates the cluster population for every cluster 
		# 'population_color':'blue',
		# 'population_alpha':1,
		# 'population_linewidth':1,
		# 'population_fontsize':10,
	},

	'fine_clustering':{
		'fine_indices_func':'func:cluster_above_mse',
		'fine_indices_func_args':[1.1],
		'clustering_scheme':[2],

		'indices_func':'func:within_cluster_weighted_err_N', 
		'indices_func_args':[],
	},

	'R_to_xyz':{
		'var_index_R':1,
		'var_index_F':2,
		'var_index_E':3,
		# needs to be in the form of R_concat 
		# (so shape = (n_samples, n_dim*n_atoms))

		'z':'func:z_from_npz_dataset',
		# is given self and dataset by default
	},

	'split_models':{
		# used for split
		'data_train_func':'func:sgdml_train_data',
		'data_train_func_args':['para:train_models,sgdml_train_args'],
		'original_indices':False, # NYI

		'classify':True,

		'mix_model':True,
		'preload_predict':False,
		'preload_predict_func':'func:sgdml_path_predict_F',
		'preload_batch_size':500,
		'preload_input_var':1,
	},

	'split_inter':{
		'branching_mode':'naive',

		'local_error_comp_index':2,
		'local_error_input_index':1,
		'local_predict_func':'func:sgdml_path_predict_F',
		'local_batch_size':500,
		'local_error_func':'func:root_mean_squared_error',

		'accept_min_size':None,
		'split_min_size':None,
		'keep_below_error':0.3,
		'checkpoints':True,
		'split_incentiviser_factor':1,
		'max_n_models':13,

		'score_function':'func:weighted_error_branch_score',
		# lower score is better
		# 
	},

}


# functions

