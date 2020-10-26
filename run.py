from util import*
import data_handling
from funcs import cluster, misc, models

def parse_arguments(args):
	'''Takes in command line arguments and outputs them as a dictionary.
	
	The main argument or "command" is handled by the parser, all subsequent
	argument are passed to the corresponding subparser. 
	
	Arguments:
		args {list}
				As given by sys.argv[:1]
	
	Returns:
		dict
				Contains all args in a condensed/formatted way
	'''

	full_call=" ".join(args)

	p=argparse.ArgumentParser(
		prog=None,
		description="Cluster dataset and create submodels for each.",
		)

	# ADD SUBPARSERS
	subp = p.add_subparsers(title='commands', dest='command', help=None)
	subp.required = True

	p_cluster = subp.add_parser(
		'cluster', 
		help='cluster dataset',
	)

	p_train = subp.add_parser(
		'train', 
		help='train improved model',
	)

	p_cluster_error = subp.add_parser(
		'cluster_error',
		help='Cluster dataset, calculate and plot errors of given model',
	)

	p_plot_cluster_error = subp.add_parser(
		'plot_error',
		help='Plot the errors for given info file',
	)

	p_cluster_xyz = subp.add_parser(
		'xyz',
		help = 'Write xyz files for every cluster'
	)

	subp_all=[p_cluster, p_train, p_cluster_error, p_cluster_xyz]

	# all except p_plot_cluster_error
	
	# ADD ARGUMENTS FOR ALL
	for x in subp_all:
		x.add_argument(
			'-d',
			'--dataset',
			metavar='<dataset_file>',
			dest='dataset_file',
			help='path to dataset file',
			required=True,
			)

		x.add_argument(
			'-p',
			'--para',
			metavar='<para_file>',
			dest='para_file',
			help="name of para file",
			required=False,
			default='default',
			)

		x.add_argument(
			'-c',
			'--cluster',
			metavar='<cluster_file>',
			dest='cluster_file',
			help='path to cluster file',
			default=None,
		   )

	# ADD ARGUMENTS FOR INFO-DEPENDENTS
	info_boys = [p_plot_cluster_error]
	for x in info_boys:
		x.add_argument(
			'-i',
			'--info',
			metavar = '<info_file>',
			dest='info_file',
			help='path to info file',
			required=True,
			)

		x.add_argument(
			'-p',
			'--para',
			metavar='<para_file>',
			dest='para_file',
			help="name of para file",
			required=False,
			default='default',
			)

	# ADD ARGUMENTS FOR RESUME
	resume_boys = []
	for x in resume_boys:
		x.add_argument(
			'-r',
			'--resume',
			metavar = '<resume_dir>',
			dest='resume_dir',
			help='path to save file from which to resume',
			required=False,
			)

	# ADD SPECIFIC ARGUMENTS FOR SUBS
	for x in [p_train]:
		x.add_argument(
			'-n',
			'--n_steps',
			metavar='<n_steps>',
			dest='n_steps',
			help='Number of steps',
			required=True,
			type = int,
			)

		x.add_argument(
			'-s',
			'--size',
			metavar='<step_size>',
			dest='step_size',
			help='Step size (in number of points)',
			required=True,
			type = int,
			)

		x.add_argument(
			'-i',
			'--init',
			metavar='<init>',
			dest='init',
			help='Initial model (path) or initial number of points (int)',
			required=True,
			)

	for x in [p_cluster_error]:
		x.add_argument(
			'-i',
			'--init',
			metavar='<init>',
			dest='init',
			help='Initial model to calculate errors of',
			required=True,
		)

	# PARSE
	args=p.parse_args()
	args=vars(args)

	# HANDLE ARGS


	# find para file
	sys.path.append("paras")
	para_file=args['para_file']

	para_file=para_file.replace(".py","") #in case the user includes '.py' in the name
	para_file=os.path.basename(para_file) 

	file=os.path.join("paras",para_file)

	if os.path.exists(file+".py"): 
		args['para_file']=para_file
	else:
		print_error(f"No valid para file found under name: {args['para_file']}")

	resume_dir = args.get('resume_dir', False)
	if resume_dir:
		args['resume'] = True
		if not os.path.exists(resume_dir):
			print_error(
				f"Tried to resume from path {resume_dir}, but doesn't exist")
	else:
		args['resume'] = False

	# add full call
	args['full_call']=full_call

	return args

class MainHandler():
	'''Main class of the program
	
	This class will contain all information needed for every operation. All 
	modules (which correspond to commands such as "cluster" or "train") inherit
	this class.
	
	Variables:
		needs_dataset {bool} 
				Default True, sometimes set to False when command/arguments 
				combination doesn't need a dataset 
		vars {list}
				List of descriptors/data in the dataset, as extracted by the 
				var funcs. See load_dataset -> var funcs in the parameter file
		info {dict}
				Information that is worth saving (for example cluster indices)
				is put in here throughout the run
		SEARCH_MODULES {list}
				See find_function method
		current_stage {number}
				Counts at which stage the program is currently at, purely for 
				ui reasons
		n_main_stages {number}
				Total number of stages, ui reasons
	'''

	def __init__(self,args, needs_dataset = True):
		'''The main use of this function is to save arguments in the object,
		determine whether this is a resumed job, load the parameters and load
		the dataset.
		
		Arguments:
			args {dict}
					Argument dictionary as returned by parse_arguments
		
		Keyword Arguments:
			needs_dataset {bool} -- (default: {True})
					Decides whether the load_dataset function gets called
		'''

		self.args=args
		self.resumed = args['resume']
		self.load_paras("default", args['para_file'])

		n_cores = int(self.call_para('n_cores') or 1)
		if n_cores==0:
			n_cores = 1
		elif n_cores<0:
			n_cores = os.cpu_count()+n_cores
		self.n_cores = n_cores

		# merge exceptions
		if self.para.get('load_dataset',{}).get('var_funcs',None) is not None:
			self.para['load_dataset']['var_funcs'] = self.para['load_dataset']['var_funcs']

		if not needs_dataset:
			self.n_main_stages -= 2 
			self.needs_dataset = False

	def load_paras(self, default, para_file):
		'''Loads the parameter file(s)
		
		Saves the combination of the default parameter file and the -p given 
		file to `self.para`. The `default.py` file is updated with the given 
		parameter file, not replaced!
		
		Arguments:
			default {[type]} -- [description]
			para_file {[type]} -- [description]
		'''

		args = self.args
		para_mod = __import__(para_file)
		para = para_mod.parameters
		funcs = func_dict_from_module(para_mod)

		# merge with defaults 
		para_def_mod = __import__(default)
		para_def = para_def_mod.parameters

		# extracts any function from the para.py file
		funcs_def = func_dict_from_module(para_def_mod)

		merge_para_dicts(para_def,para) #WAI
		# make an exception for certain things
		if ('load_dataset' in para) and ('var_funcs' in para['load_dataset']):
			para_def['load_dataset']['var_funcs'] = \
				para['load_dataset']['var_funcs']

		self.para = para_def
		z = {**funcs_def,**funcs} #WAI
		self.funcs = z

	needs_dataset = True
	vars=[]
	info = {}

	SEARCH_MODULES=[cluster, data_handling, misc, models]
	def find_function(self,name):
		'''Finds a function of a given variable name
		
		This method looks through all modules given in SEARCH_MODULES (in order)
		and searches a function of the given name. If found, it returns a
		pointer to the function.
		
		Arguments:
			name {string} 
					Name of the function
		
		Returns:
			[function] or [None]
					Returns the function or None if not found
		'''
		return find_function(name,self.funcs,self.SEARCH_MODULES)

	def generate_para_args(self,args):
		'''Dummy function, see generate_custom_args in the `utils.py` file'''
		return generate_custom_args(self,args)

	def generate_para_kwargs(self, kwargs):
		'''Dummy function, see generate_custom_kwargs in the `utils.py` file'''
		return generate_custom_kwargs(self, kwargs)

	def call_para(self,*path,args=[], kwargs={}):
		'''Given a path in the parameter file, calls the parameter if callable,
		otherwise return it.
		
		If a not-None input is found under the given parameter path, we check 
		if it contains a `func:` prefix. If yes, the find_function function is 
		called, which returns a pointer to the function of the same name as 
		given in the parameter file. The function is then called with the args
		and kwargs passed to the function as well as those contained in the 
		`*_args` parameter (where * is the function name).
		If the parameter does not contain a `func:` prefix, the parameter itself 
		is simply returned.
		
		Arguments:
			*path {list}
					List of steps to take to reach the given parameter. For 
					example: ['clusters', 0, 'n_clusters']
		
		Keyword Arguments:
			args {list} (default: {[]})
					List of args as passed by the program
			kwargs {dict} (default: {{}})
					List of kwargs as passed by the program
		
		Returns:
			[/]
					Returns None if no function is found under the given name,
					otherwise returns the outputs of the function or the 
					parameter itself if no `func:` prefix was present
		'''
		if len(path)==0:
			return None

		para=self.para
		subdict=para
		step=para


		for x in path:
			subdict=step
			step=step.get(x,None)


		# handle functions
		if type(step)==str and step.startswith("func:"):
			f_name=step.replace("func:","")
			f=self.find_function(f_name)

			arg_name=str(path[-1])+'_args'
			args_add=self.generate_para_args(subdict.get(arg_name,[]))

			kwarg_name=str(path[-1])+'_kwargs'
			kwargs_add=self.generate_para_kwargs(subdict.get(kwarg_name,{}))

			kwargs.update(kwargs_add)

			args_full=args+args_add
			return f(*args_full, **kwargs)

		elif type(step)==str and step.startswith("para:"):
			new_args=step.replace("para:","").split(",")
			return self.call_para(*newargs,args=args)

		elif step is None:
			return None
		else:
			# Not needed any more, call_para is more versatile and so is the default now
			# print_warning(f"Tried to call para: {path}={step}, but not callable. Value returned instead.")
			return step

	def return_partial_func(self, *path, kwargs = {}):
		if len(path)==0:
			return None

		para=self.para
		subdict=para
		step=para

		for x in path:
			subdict=step
			step=step.get(x,None)


		# handle functions
		if type(step)==str and step.startswith("func:"):
			f_name=step.replace("func:","")
			f=self.find_function(f_name)
			kwarg_name=str(path[-1])+'_kwargs'
			kwargs_add=self.generate_para_kwargs(subdict.get(kwarg_name,{}))
			kwargs.update(kwargs_add)
		else:
			print_error(f'Para {path} not a function')

		func = self.get_para(*path)
		return partial(func, **kwargs)

	def get_para(self,*path,args=[]):
		'''See call_para. Same idea except it returns a pointer to the function
		if a function is found (with the `func:` prefix)'''
		if len(path)==0:
			return None

		para=self.para
		step=para

		for x in path:
			step=step.get(x,None)

		# handle functions
		if type(step)==str and step.startswith("func:"):
			f_name=step.replace("func:","")
			f=self.find_function(f_name)
			return f
		elif type(step)==str and step.startswith("para:"):
			new_args=step.replace("para:","").split(",")
			return self.get_para(*newargs,args=args)
		else:
			return step

	def print_stage(self, s):
		'''Prints the current stage in the UI
		
		Arguments:
			s {string}
					Title of the stage
		'''
		print_stage(s, self.current_stage, self.n_stages)
		self.current_stage += 1

	current_stage = 1 
	n_main_stages = 4
	def run(self):
		'''
		Runs the program. The dataset is loaded, the vars and storage are 
		prepared and then the `.run_command` method is called, which is defined
		in the called module (for example: `cluster` or `train`)

		At the end, information is saved and temporary files are deleted.
		'''

		if self.needs_dataset:
			self.print_stage('Load dataset')
			self.load_dataset()

			self.print_stage('Prepare vars')
			self.prepare_vars()

		self.print_stage('Prepare storage')
		if self.args['resume']:
			self.resume_storage()
			self.resume_command()

		else:
			self.prepare_storage()
			self.run_command()

		self.print_stage('Save in storage')
		self.save_main()
		self.save_command()
		self.delete_temp()

	def delete_temp(self):
		'''
		Deletes all temporary files from storage
		'''
		if self.call_para('remove_temp_files'):
			shutil.rmtree(self.temp_dir)

	def load_dataset(self):
		''' Loads the dataset and stores it in `.dataset`

		Currently supported:
		xyz format - needs to be extended with energy in the comment and forces
		npz format - as given by sGDML, needs to contain 'R', 'E', 'F'
		db  format - as given by schnetpack
	   
		'''
		path=self.args['dataset_file']

		if path is None:
			print_error(f"No dataset given. Please use the -d arg followed by the path to the dataset.")
		elif not os.path.exists(path):
			print_error(f"Dataset path {path} is not valid.")

		ext=os.path.splitext(path)[-1]
		#xyz file
		if ext==".xyz":
			print_ongoing_process(f"Loading xyz file {path}")
			try:
				file=open(path)
				dat=read_concat_ext_xyz(file)
				data={ 'R':np.array(dat[0]),'z':dat[1],'E':np.reshape( dat[2] , (len(dat[2]),1) ),'F':np.array(dat[3]) }
			except Exception as e:
				print(e)
				print_error("Couldn't load .xyz file.")

			print_ongoing_process(f"Loaded xyz file {path}",True)

		#npz file        
		elif ext==".npz":
			print_ongoing_process(f"Loading npz file {path}")
			try:
				data=np.load(path,allow_pickle=True)
			except Exception as e:
				print(e)
				print_error("Couldn't load .npz file.")

			print_ongoing_process(f"Loaded npz file {path}",True)

		# schnetpack .db
		elif ext == '.db':
			print_ongoing_process(f"Loading db file {path}")

			from schnetpack import AtomsData
			data = AtomsData(path)

			print_ongoing_process(f"Loaded db file {path}", True)


		else:
			print_error(f"Unsupported data type {ext} for given dataset {path} (xyz, npz, schnetpack .db supported).")

		
		self.dataset=data
		self.dataset_path = path
		if self.get_para('load_dataset','post_processing') is not None:
			print_ongoing_process('Post-processing dataset')
			self.call_para('load_dataset','post_processing',args=[self])
			print_ongoing_process('Post-processing dataset',True)

	def prepare_vars(self):
		'''Prepares the descriptors/information in the dataset
		
		Loop through every function given in the load_dataset->var_funcs 
		parameter file and call them with the dataset as an argument. The 
		output of those functions is then saved in `self.vars[i]` where i is 
		the index of the function in the var_funcs parameter list
		'''

		dataset=self.dataset

		#get the needed vars ready
		#parses through data set and uses the given functions to generate the needed variables
		#f.e. interatomic distances and energies
		var_funcs=self.call_para('load_dataset','var_funcs')
		keys=list(var_funcs.keys())
		for i in range(len(keys)):
			print_x_out_of_y("Extracting vars",i,len(keys))
			x=keys[i]
			self.vars.append(self.call_para("load_dataset","var_funcs",x,args=[self, self.dataset]))
		print_x_out_of_y("Extracting vars",len(keys),len(keys),True)

		# SUMMARY
		summary_table={}
		for i in range(len(self.vars)):
			try:
				summary_table[i]=self.vars[i].shape
			except:
				summary_table[i]="No shape"
		print_table("Vars summary:","index","shape",summary_table)

	def do_nothing(*args):
		'''Useless dummy/debug method'''
		print_debug("Doing nothing. Please be patient.")

	def resume_storage(self):
		'''When a task is resumed, re-use the same storage and check what it 
		contains
		'''
		args = self.args
		path = args['resume_dir']
		self.storage_dir = path
		self.temp_dir = os.path.join(path, 'temp')

		if os.path.exists(path):
			print_ongoing_process(f"Save path {path} found.", True)

		cp_path = os.path.join(path, 'checkpoint.p')
		if not os.path.exists(cp_path):
			print_error(f"No checkpoint file found at {cp_path}")

		with open(cp_path, 'rb') as file:
			info = pickle.loads(file.read())

		self.resume_info = info 

		if self.call_para('storage','save_original_call'):
			print_ongoing_process('Saving call')
			with open(os.path.join(path,"Call.txt"),'a+') as file:
				print(f"Resume call: {args.get('full_call','N/A')}",file=file)
				print_ongoing_process(f'Saved resume call in {os.path.join(path,"Call.txt")}',True)
		
		# NOT YET SUPPORTED
		# print_ongoing_process("Searching for parameter files")
		# def_para = "default"
		# add_para = args['para_file']
		# for file in glob.glob(os.path.join(path,'*.py')):
		#     print(file)
		#     if file.startswith("default"):
		#         def_para = file
		#     else:
		#         add_para = file

		# self.load_paras(def_para, add_para)
		# print_ongoing_process("Searching for parameter files", True)

	def prepare_storage(self):
		'''
		Prepares the storage directory. By default, the name of the storage is 
		"{command_name}_{basename_of_dataset}" and is saved inside the "saves/"
		folder. 
		'''
		print_ongoing_process("Preparing save directory")
		storage_dir=self.call_para('storage','storage_dir')
		dir_name=f"{self.args['command']}_{self.call_para('storage','dir_name')}"

		path=find_valid_path(os.path.join(storage_dir,dir_name))
		self.storage_dir=path

		if not os.path.exists(path):
			os.makedirs(path)
		else:
			print_warning(f"Save path {path} already exists. How? Overwriting of files possible.")
		print_ongoing_process(f"Prepared save directory {path}",True)

		# copy user para file
		if self.call_para('storage','save_para_user'):
			print_ongoing_process("Saving user para file")
			file_name=self.args.get("para_file")+".py"
			file=os.path.join("paras",file_name)
			if os.path.exists(file):
				shutil.copy(file,os.path.join(path,file_name))
				print_ongoing_process(f"Saved user para file {os.path.join(path,file_name)}",True)
			else:
				print_warning(f"Tried copying user parameter file {file}. Not found")

		# copy default para file
		if self.call_para('storage','save_para_default'):
			print_ongoing_process('Saving default para file')
			file_name="default.py"
			file=os.path.join("paras",file_name)
			if os.path.exists(file):
				shutil.copy(file,os.path.join(path,file_name))
				print_ongoing_process(f'Saved default para file {os.path.join(path,file_name)}',True)
			else:
				print_warning(f"Tried copying default parameter file {file}. Not found")

		if self.call_para('storage','save_original_call'):
			print_ongoing_process('Saving original call')
			with open(os.path.join(path,"Call.txt"),'w+') as file:
				print(f"Original call: {self.args.get('full_call','N/A')}",file=file)
				print_ongoing_process(f'Saved original call at {os.path.join(path,"Call.txt")}',True)

		# create temp folder
		self.temp_dir = os.path.join(self.storage_dir, 'temp')
		os.mkdir(self.temp_dir)

	def save_main(self):
		'''The main saving function that every command goes through
		
		By default, pickles and saves the `.info` dictionary only (all other 
		things are saved by the modules corresponding to the chosen command)
		'''
		self.info['para'] = self.para
		self.info['args'] = self.args

		print_ongoing_process('Saving info file')
		info_path = os.path.join(self.storage_dir,'info.p')
		with open(info_path,'wb') as file:
			pickle.dump(self.info,file)
		print_ongoing_process('Saved info file', True)

	def load_info_file(self, path):
		'''
		Loads the info file, used for resuming tasks
		'''
		print_ongoing_process('Loading info file')
		with open(path,'rb') as file:
			info = pickle.loads(file.read())

		self.info = info
		if 'cluster_indices' in info:
			self.cluster_indices = info['cluster_indices']

		if 'errors' in info:
			self.errors = info['errors']

		info['args'] = self.args
		print_ongoing_process('Loaded info file', True)
		summary_table = {}
		for k,v in info.items():
			summary_table[k] = f'{type(v)}'
		print_table("Items found:","Key","Value",summary_table, width = 22)

if __name__=='__main__':

	args=parse_arguments(sys.argv[1:])
	command = args['command']

	if command == 'cluster':
		from modules.cluster import ClusterHandler 
		hand = ClusterHandler(args)

	elif command == 'train':
		from modules.train import TrainHandler 
		hand = TrainHandler(args)

	elif command == 'cluster_error':
		from modules.clustererror import ClusterErrorHandler 
		hand = ClusterErrorHandler(args)

	elif command == 'plot_error':
		from modules.plotclustererror import PlotClusterErrorHandler 
		hand = PlotClusterErrorHandler(args, needs_dataset = False)

	elif command == 'xyz':
		from modules.cluster_xyz import ClusterXYZHandler
		hand = ClusterXYZHandler(args)

	# actually run the friggin' thing
	hand.run()

	print_successful_exit("run.py exited successfully")