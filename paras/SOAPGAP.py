gap_base_command = """
export PATH=/home/users/gfonseca/QUIP/build/linux_x86_64_gfortran/:$PATH

module load lang/SciPy-bundle/2019.03-intel-2019a
module load tools/binutils/2.32-GCCcore-8.2.0

source /home/users/gfonseca/anaconda3/bin/activate quippy

cd __WORK_DIR__

gap_fit at_file=__TRAIN_FILE__ \
    gap={soap cutoff=5.0 \
        covariance_type=dot_product \
        zeta=2 \
        delta=0.25 \
        atom_sigma=0.3 \
        add_species=T \
        n_max=12 \
        l_max=6 \
        n_sparse=4000 \
        sparse_method=cur_points} \
    force_parameter_name=forces \
    e0_method=average \
    default_sigma={0.001 0.2 0.0 0.0} \
    do_copy_at_file=F sparse_separate_file=F \
    gp_file=__END_FILE__

echo Donezo
wait
"""

parameters = {
    "load_dataset": {
        "post_processing": None,
        "post_processing_args": [],
        "schnet_preprocessed_npz": False,
        "var_funcs": {
            0: "func:r_to_dist",
            1: "func:npz_indices",
            2: "func:extract_F_concat",
            3: "func:extract_E",
        },  # end of 'var_funcs'
    },
    "clusters": {
        "save_npy": True,
        "init_scheme": [0, 1],
        # indices below define the clustering scheme
        0: {
            "type": "func:agglomerative_clustering",
            "n_clusters": 10,
            "initial_number": 20000,
            "custom": False,  ## distance_matrix_func and linkage only active if True
            "distance_matrix_function": "func:distance_matrix_euclidean",
            "linkage": "complete",
            "cluster_choice_criterion": "func:smallest_max_distance_euclidean",
            "var_index": 0,
        },
        1: {
            "type": "func:kmeans_clustering",
            "n_clusters": 5,
            "var_index": 3,
        },
        2: {
            "type": "func:agglomerative_clustering",
            "n_clusters": 100,
            "initial_number": 10000,
            "custom": False,
            "distance_matrix_function": "func:distance_matrix_euclidean",
            "linkage": "complete",
            "cluster_choice_criterion": "func:smallest_max_distance_euclidean",
            "var_index": 0,
        },
    },  # end of 'clusters'
    "train_models": {
        #
        "model_ext": "",  #  is a directory, in order to save trind
        "train_func": "func:gapsoap_train_default",
        "train_func_args": [gap_base_command],
        "load_func": "func:load_gap_model",  # needs to also load trianing set
        "model_info_func": None,
    },
    "predict_error": {
        "error_sub_indices": "func:return_second_argument",  # twenty_each
        "main_error_index": 0,
        #  the error that is used for those commands that only need one type
        #  of error (like 'train') rather than an entire analysis
        #  error will be saved into self.cluster_err (by_cluster must be true!)
        # only main error is calculated if extended = False in calculate_error
        "sample_wise_error_func": "func:MSE_sample_wise",
        "sample_wise_predicts_index": 0,
        "sample_wise_comparison_var_index": 2,
        #  sample-wise only relevant if sample_wise = True in calculate_error
        "predicts": {
            0: {
                "predict_func": "func:gap_predict_F",
                "input_var_index": 1,
            },
        },  # end of predicts
        # RMSE
        0: {
            "predicts_index": 0,
            "comparison_var_index": 2,
            "by_cluster": True,
            "error_func": "func:root_mean_squared_error",
            "file_name": "RMSE_graph",
            "save_key": "RMSE_c",  # will be saved in save.p under the given key (in 'errors' sub dict)
            "graph": True,
            "graph_paras": {
                "y_axis_label": r"Force prediction error $(kcal/mol\,\AA)$",
                "real_average_key": "RMSE_o"
                # replace default paras here
            },
        },
        # overall RMSE
        1: {
            "predicts_index": 0,
            "comparison_var_index": 2,
            "error_func": "func:root_mean_squared_error",
            "save_key": "RMSE_o",  # will be saved in save.p under the given key (in 'errors' sub dict)
            "graph": False,
            "graph_paras": {},
        },
        # MAE
        2: {
            "predicts_index": 0,
            "comparison_var_index": 2,
            "by_cluster": True,
            "error_func": "func:mean_absolute_error",
            "file_name": "MAE_graph",
            "save_key": "MAE_c",  # will be saved in save.p under the given key (in 'errors' sub dict)
            "graph": True,
            "graph_paras": {
                "y_axis_label": r"Force prediction error $(kcal/mol\,\AA)$",
                "real_average_key": "MAE_o"
                # replace default paras here
            },
        },
        # overall MAE
        3: {
            "predicts_index": 0,
            "comparison_var_index": 2,
            "error_func": "func:mean_absolute_error",
            "save_key": "MAE_o",  # will be saved in save.p under the given key (in 'errors' sub dict)
            "graph": False,
            "graph_paras": {},
        },
        # MSE
        4: {
            "predicts_index": 0,
            "comparison_var_index": 2,
            "by_cluster": True,
            "error_func": "func:mean_squared_error",
            "file_name": "MSE_graph",
            "save_key": "MSE_c",  # will be saved in save.p under the given key (in 'errors' sub dict)
            "graph": True,
            "graph_paras": {
                "y_axis_label": r"Force prediction error $(kcal/mol\,\AA)^2$",
                "real_average_key": "MSE_o",
            },
        },
        # overall MSE
        5: {
            "predicts_index": 0,
            "comparison_var_index": 2,
            "error_func": "func:mean_squared_error",
            "save_key": "MSE_o",  # will be saved in save.p under the given key (in 'errors' sub dict)
            "graph": False,
            "graph_paras": {},
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
}


#  functions
