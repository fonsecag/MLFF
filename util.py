import subprocess,pickle,os,sys,getopt,shutil,logging,glob,re
import math,joblib,argparse,copy,time,contextlib,functools,itertools
import numpy as np
import threading
from types import ModuleType 
from functools import partial 

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
logging.disable(logging.DEBUG)

UI_COLUMN_WIDTH = 10
SEPARATOR_CHARACTER_1 = "#"

#UNBUFFER PRINT
_print=functools.partial(print, flush=True)

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	FLASH = '\033[5m'
	UNDERLINE = '\033[4m'
	LIGHTGREY = '\033[37m'
	BGBLACK='\033[40m'
	WHITE='\033[97m'

def print_stage(s,a,b):
	_print("\n"+bcolors.LIGHTGREY + bcolors.BGBLACK
		+f"STAGE {a}/{b} " 
		+ bcolors.ENDC + bcolors.BOLD 
		+ str(s) 
		+ bcolors.ENDC)
	_print(SEPARATOR_CHARACTER_1*70)

def print_x_out_of_y(s,x,y,finish=False,width=None):
	a=f"({x}/{y})"
	if width is None:
		width=UI_COLUMN_WIDTH
	if finish:
		_print(f"{bcolors.OKGREEN}{a:<{width}}{bcolors.ENDC}{s}       ")
	else:
		_print(f"{bcolors.OKBLUE}{a:<{width}}{bcolors.ENDC}{s}{bcolors.FLASH}...{bcolors.ENDC}",end="\r")

def print_x_out_of_y_eta(s,x,y,eta,finish=False,width=None):
	a=f"({x}/{y})"
	if width is None:
		width=UI_COLUMN_WIDTH
	if finish:
		eta=f"({eta:.1f}s)"
		_print(f"{bcolors.OKGREEN}{a:<{width}}{bcolors.ENDC}{s:<70}{bcolors.WHITE}{eta:>10}{bcolors.ENDC}")
	else:
		rem_len=70-len(s)
		eta=f"(ETA {eta:.1f}s)"
		_print(f"{bcolors.OKBLUE}{a:<{width}}{bcolors.ENDC}{s}"
			+f"{bcolors.FLASH}{'...':<{rem_len}}{bcolors.ENDC}{bcolors.WHITE}{eta:>10}{bcolors.ENDC}",end="\r")

def print_subtitle(name):
	_print(f"\n{bcolors.LIGHTGREY}{bcolors.UNDERLINE}{name}{bcolors.ENDC}")

def print_percent(s,x,y,finish=False):
	p=int(float(x)/y*100)+1
	print_percent_last_value_printed=p
	if finish:
		_print(f"{bcolors.OKGREEN}{'(100%)':<{UI_COLUMN_WIDTH}}" \
			f"{bcolors.ENDC}{s}       ")
	else:
		a=f'({p}%)'
		_print(f"{bcolors.OKBLUE}{a:<{UI_COLUMN_WIDTH}}{bcolors.ENDC}{s}" \
			f"{bcolors.FLASH}...{bcolors.ENDC}",end="\r")

class sgdml_print_suppressor:
	# note: doesnt supporess EVERYTHING
	# specifically, not the things about reuising initial model...
	# Honestly no god damn clue why
	def __enter__(self):
		self.sgdml_level=logging.getLogger('sgdml').getEffectiveLevel()
		self.sgdml_cli_level=logging.getLogger('sgdml.cli').getEffectiveLevel()
		logging.getLogger('sgdml').setLevel(logging.ERROR)
		logging.getLogger('sgdml.cli').setLevel(logging.ERROR)
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		logging.getLogger('sgdml').setLevel(self.sgdml_level)
		logging.getLogger('sgdml.cli').setLevel(self.sgdml_cli_level)
		sys.stdout.close()
		sys.stdout = self._original_stdout
		
	sgdml_level='DEBUG'
	sgdml_cli_level='DEBUG'

def print_ongoing_process(s,finish=False,time=None):
	if finish:
		if time is not None:
			time_s=f"({time:.1f}s)"
		else:
			time_s=""
		_print(f"{bcolors.OKGREEN}{'(DONE)':<{UI_COLUMN_WIDTH}}{bcolors.ENDC}"
			+f"{s:<70}{bcolors.WHITE}{time_s:>10}{bcolors.ENDC}")
	else:
		_print(f"{bcolors.OKBLUE}{'(WAIT)':<{UI_COLUMN_WIDTH}}{bcolors.ENDC}"
			+f"{s}{bcolors.FLASH}...{bcolors.ENDC}{' '*20}",end="\r")

def print_cluster_step(a,b):
	_print(f"\n{bcolors.LIGHTGREY}{bcolors.UNDERLINE}Cluster step {a}/{b}{bcolors.ENDC}")

def print_cluster_scheme(scheme,para):
	_print(f"\n{bcolors.UNDERLINE}{bcolors.LIGHTGREY}{'Clustering scheme'}{bcolors.ENDC}")

	_print(f"{'Index':<7}{'clusters':<10}{'type'}")

	for i in scheme:
		info=para[i]
		_print(f"{i:<7}{info.get('n_clusters','N/A'):<10}{info.get('type','N/A').replace('func:','')}")

def print_table(name,c1,c2,table,width=None):
	if width is None:
		width=UI_COLUMN_WIDTH
	print_subtitle(name)
	if not (c1 is None):
		_print(f"{c1:<{width}}{c2}")
	for k,v in table.items():
		_print(f"{k:<{width}}{v}")

def print_warning(s):
	_print(f"\n{bcolors.WARNING}{'(WARN)':<{UI_COLUMN_WIDTH}}{bcolors.ENDC}{str(s)}")

def print_info(s):
	_print(f"{bcolors.WHITE}{'(INFO)':<{UI_COLUMN_WIDTH}}{bcolors.ENDC}{str(s)}")

def print_error(s):
	_print(f"{bcolors.FAIL}{'(ERR)':<{UI_COLUMN_WIDTH}}{str(s)}{bcolors.ENDC}")
	sys.exit(2)

def print_blue(s):
	_print(bcolors.OKBLUE + str(s) + bcolors.ENDC)

def print_debug(s):
	_print(f"\n{bcolors.OKGREEN}{str(s)}{bcolors.ENDC}")

def print_successful_exit(s):
	happy_man=' \\o/  '
	_print(f"\n{bcolors.OKGREEN}{happy_man:<{UI_COLUMN_WIDTH}}{bcolors.ENDC}{s}")

def find_function(name,para,SEARCH_MODULES):
	if name is None:
		print_error('Error in find_function: name is None. Aborted.')

	#deprecated, returns a function that returns the int/float 
	if isinstance(name,float) or isinstance(name,int):
		return lambda: name
		
	if not isinstance(name,str):
		print_error('Error in find_function: {} not a string. Aborted.'.format(name))

	search=SEARCH_MODULES+[para]
	for a in search:
		if isinstance(a,dict):
			f=((name in a) and a[name]) or False
			if callable(f):
				return f

		elif isinstance(a,ModuleType):
			f=getattr(a,name,False)
			if callable(f):
				return f

	print_error('No callable function called {} found in find_function. Aborted.'.format(name))
	
def merge_para_dicts(default,user):
	for k,v in user.items():
		if type(v)==dict:
			if not (k in default):
				default[k]=copy.deepcopy(v)
			else:
				merge_para_dicts(default[k],v)

		else:
			default[k]=copy.copy(v)

def generate_custom_args(self,args):
	
	if not (isinstance(args,list) or isinstance(args,tuple)):
		print_error('Invalid variable type for args in generate_custom_args. Must be tuple or list. Aborted')

	a=[]
	for i in args:
		if isinstance(i,str) and i.startswith('func:'):
			f_name=i.replace('func:','')
			f=find_function(f_name,self.funcs,self.SEARCH_MODULES)
			b=f(self)
			a.append(f(self))

		elif isinstance(i,str) and i.startswith('para:'):
			new_args=i.replace("para:","").split(",")
			try:
				b=self.call_para(*new_args,args=[self])
			except Exception as e:
				b=None
			a.append(b)

		else:
			a.append(i)

	return a

def generate_custom_kwargs(self,kwargs):
	
	if not (isinstance(kwargs,dict)):
		print_error('Invalid variable type for kwargs in generate_custom_kwargs. Must be dict.')

	a={}
	for k,v in kwargs.items():
		if isinstance(i,str) and i.startswith('func:'):
			f_name=i.replace('func:','')
			f=find_function(f_name,self.funcs,self.SEARCH_MODULES)
			b=f(self)
			a.append(f(self))

		elif isinstance(i,str) and i.startswith('para:'):
			new_args=i.replace("para:","").split(",")
			try:
				b=self.call_para(*new_args, args=[self])
			except Exception as e:
				b=None
			a.append(b)

		else:
			a.append(i)

	return a

def run_custom_file(db,file,args):
	file_name,ext=os.path.splitext(file)
	path=custom_files_dir+file

	args=generate_custom_args(db,args)

	if ext=='.py':
		file=__import__(file_name)
		file.run(*args)
	elif ext=='.sh':
		call=[path]+[str(x) for x in args] #bash script needs it all to be strings ofc
		subprocess.call(call)
	else:
		print_error("File type {} not recognised in run_custom_file. Aborted.".format(ext))

def find_valid_path(path):
	n,ori=1,path
	while os.path.exists(path):
		path='{}_{}'.format(ori,n)
		n+=1
		if n>1000:
			print_error("Broke out of 'find_valid_path' after {} iterations (current path: {}). How?".format(n,path))
	return path

def func_dict_from_module(mod):
	context = {}
	for setting in dir(mod):
		a = getattr(mod, setting)
		if callable(a):
			context[setting]=a
	return context

def color_interp(color_steps, values):
	from scipy.interpolate import interp1d

	y_c = np.array(color_steps)
	n_color_steps = len(color_steps)
	x_min, x_max = np.min(values), np.max(values)
	delta = x_max - x_min

	x_c = [  x_min + delta * j / (n_color_steps-1)
		for j in range(n_color_steps)  ]
	# sets the color for intermediate color points linearly

	f_c = interp1d(x_c, y_c, axis=0)

	bar_color=f_c(values)

	return bar_color