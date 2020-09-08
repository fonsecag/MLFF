from util import*

def spl_return_self(self):
	return self

def get_dataset_basename(self):
	file=self.args.get('dataset_file','UNKNOWN')
	return os.path.basename(file).split('.')[0]

def get_splitter_cluster_indices_i(self,i):
	return self.cluster_indices[i]

def get_splitter_n_clusters(self):
	return len(self.cluster_indices)

def save_subdata_sgdml_i(self,i):
	from sgdml.utils.io import dataset_md5

	ind=self.subdata_indices[i]
	name=os.path.join(self.subdata_path,f"subdata_{i}.npz")

	data=dict(self.dataset)
	for i in ['R','E','F']:
		data[i]=data[i][ind]
	data['name']=name
	data['md5']=dataset_md5(data)

	np.savez_compressed(name,**data)

def MSE_sample_wise(self, x,y):
	err=(np.array(x)-np.array(y))**2
	return err.mean(axis=1)

def return_second_argument(a, b):
	return b

def twenty_each(self, ind):
	a = []
	for x in ind:
		a.append(np.random.choice(x, 20, replace = False))

	return a

def mean_squared_error(self, x,y):
	# print(f'{np.average(x)} {np.average(y)} mean_squared_error')
	from sklearn.metrics import mean_squared_error as MSE
	err=MSE(np.array(x),np.array(y))
	return err

def mean_absolute_error(self, x,y):
	# print(f'{np.average(x)} {np.average(y)} mean_absolute_error')
	from sklearn.metrics import mean_absolute_error as MAE
	err=MAE(np.array(x),np.array(y))
	return err

def root_mean_squared_error(self, x,y):
	# print(f'{np.average(x)} {np.average(y)} root_mean_squared_error')
	from sklearn.metrics import mean_squared_error as MSE

	err=MSE(np.array(x),np.array(y))
	return np.sqrt(err)

def root_mean_squared_error_atom_wise(self, x,y):
	# print(f'{np.average(x)} {np.average(y)} root_mean_squared_error')
	from sklearn.metrics import mean_squared_error as MSE
	n_samples = x.shape[0]
	a, b = np.reshape(x, (n_samples, -1, 3)), np.reshape(y, (n_samples, -1, 3))
	n_atoms = a.shape[1]
	err= [ MSE(  a[:,i,:], b[:,i,:]) for i in range(n_atoms)]
	return np.sqrt(err)

def root_mean_squared_error_atom_wise_bins(self, x,y):
	from sklearn.metrics import mean_squared_error as MSE
	n_samples = x.shape[0]
	x, y = np.reshape(x, (n_samples, -1, 3)), np.reshape(y, (n_samples, -1, 3))
	n_atoms = x.shape[1]


	F = self.dataset['F']
	ranges = [0, .5, 1, 1.5, 2]
	F_norm = np.linalg.norm(F, axis = 2)

	atom_errors = []
	for i in range(0, n_atoms):
	    atom_errors.append([])
	    F_norm_atom = F_norm[:,i]
	    std_atom, mean_atom = np.std(F_norm_atom), np.mean(F_norm_atom)

	    for j in range(len(ranges)):
	        a = ranges[j]
	        if j == len(ranges) - 1:
	            b = np.inf 
	        else:
	            b = ranges[j + 1]
	 
	        a, b = std_atom*a, std_atom*b
	        F_renorm = np.abs(F_norm_atom -  mean_atom)
	        ind = np.argwhere(  np.logical_and(F_renorm>=a, F_renorm<=b))
	        ind = ind.reshape(-1)

	        atom_errors[i].append( np.sqrt(MSE( x[ind,i,:], y[ind,i,:])))

	return atom_errors



_z_str_to_z_dict = {
	'H':1,
	'He':2,
	'Li':3,
	'Be':4,
	'B':5,
	'C':6,
	'N':7,
	'O':8,
	'F':9,
	'Ne':10,
	'Na':11,
	'Mg':12,
	'Al':13,
	'Si':14,
	'P':15,
	'S':16,
	'Cl':17,
	'Ar':18,
	'K':19,
	'Ca':20,
	'Sc':21,
	'Ti':22,
	'V':23,
	'Cr':24,
	'Mn':25,
	'Fe':26,
	'Co':27,
	'Ni':28,
	'Cu':29,
	'Zn':30,
	'Ga':31,
	'Ge':32,
	'As':33,
	'Se':34,
	'Br':35,
	'Kr':36,
	'Rb':37,
	'Sr':38,
	'Y':39,
	'Zr':40,
	'Nb':41,
	'Mo':42,
	'Tc':43,
	'Ru':44,
	'Rh':45,
	'Pd':46,
	'Ag':47,
	'Cd':48,
	'In':49,
	'Sn':50,
	'Sb':51,
	'Te':52,
	'I':53,
	'Xe':54,
	'Cs':55,
	'Ba':56,
	'La':57,
	'Ce':58,
	'Pr':59,
	'Nd':60,
	'Pm':61,
	'Sm':62,
	'Eu':63,
	'Gd':64,
	'Tb':65,
	'Dy':66,
	'Ho':67,
	'Er':68,
	'Tm':69,
	'Yb':70,
	'Lu':71,
	'Hf':72,
	'Ta':73,
	'W':74,
	'Re':75,
	'Os':76,
	'Ir':77,
	'Pt':78,
	'Au':79,
	'Hg':80,
	'Tl':81,
	'Pb':82,
	'Bi':83,
	'Po':84,
	'At':85,
	'Rn':86,
	'Fr':87,
	'Ra':88,
	'Ac':89,
	'Th':90,
	'Pa':91,
	'U':92,
	'Np':93,
	'Pu':94,
	'Am':95,
	'Cm':96,
	'Bk':97,
	'Cf':98,
	'Es':99,
	'Fm':100,
	'Md':101,
	'No':102,
	'Lr':103,
	'Rf':104,
	'Db':105,
	'Sg':106,
	'Bh':107,
	'Hs':108,
	'Mt':109,
	'Ds':110,
	'Rg':111,
	'Cn':112,
	'Uuq':114,
	'Uuh':116
}
_z_to_z_str_dict = {v: k for k, v in _z_str_to_z_dict.items()}

def z_from_npz_dataset(self, dataset):
	z = dataset['z']
	z_n = [_z_to_z_str_dict[x] for x in z]
	return z_n

def weighted_error_branch_score(self, branches):

	print('\n')
	print(f"{'Branch':<7}{'avg err':<9}{'size'}")

	n_tot, err_pop = 0, 0
	for i in range(len(branches)):
		branch = branches[i]
		err, pop = branch['err'], len(branch['ind'])
		err_pop += err*pop
		n_tot += pop
		print(f"{i:<7}{err:<9.2f}{pop}")

	return err_pop/n_tot
