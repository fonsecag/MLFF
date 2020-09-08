from run import MainHandler
from .cluster import ClusterHandler
from util import*


class ClusterXYZHandler(ClusterHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)
		self.n_stages = self.n_main_stages + self.n_substages

	n_substages = ClusterHandler.n_substages   # one more than ClusterHandler

	def run_command(self):
		super().run_command()

	def save_cluster_xyz(self):
		
		z = self.call_para('R_to_xyz', 'z',
			args = [self, self.dataset]
			)
		self.z = z

		dir_path = os.path.join(self.storage_dir, 'cluster_xyz')
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)

		var_index_R = self.call_para('R_to_xyz', 'var_index_R')
		R = self.vars[var_index_R]

		var_index_F = self.call_para('R_to_xyz', 'var_index_F')
		F = self.vars[var_index_F]

		var_index_E = self.call_para('R_to_xyz', 'var_index_E')
		E = self.vars[var_index_E]

		cl_ind = self.cluster_indices

		for i in range(len(cl_ind)):
			cl = np.array(cl_ind[i], dtype = np.int64)
			self.save_xyz_index(i, R[cl], F[cl], E[cl])

	# def save_cluster_xyz(self):
		# multithreading seemed pointless in the end because even though we're
		# writing into files here, the entire task does not seem to be i/o bound
		# at all, as performance with or without threads was essentially same
		
	# 	z = self.call_para('R_to_xyz', 'z',
	# 		args = [self, self.dataset]
	# 		)
	# 	self.z = z

	# 	dir_path = os.path.join(self.storage_dir, 'cluster_xyz')
	# 	if not os.path.exists(dir_path):
	# 		os.mkdir(dir_path)

	# 	var_index = self.call_para('R_to_xyz', 'var_index')
	# 	R = self.vars[var_index]
	# 	self.cl_xyz_threads = []

	# 	cl_ind = self.cluster_indices
	# 	for i in range(len(cl_ind)):
	# 		cl = cl_ind[i]
	# 		thr = threading.Thread(target = self.save_xyz_index,
	# 			args = (i, R[cl]))
	# 		thr.start()
	# 		self.cl_xyz_threads.append(thr)
	# 		# self.save_xyz_index(i, R[cl])


	# 	for x in self.cl_xyz_threads:
	# 		x.join()

	def save_xyz_index(self, i, R, F, E):
		file_name = f'cluster_{i}.xyz'
		path = os.path.join(self.storage_dir, 'cluster_xyz', file_name)


		file = open(path, 'w+')
		for j in range(len(R)):
			r_j, f_j, e_j = R[j], F[j], E[j]
			s = self.RFE_to_xyz_single(r_j, f_j, e_j)
			file.write(s)

		file.close()

# Energy=-620726.002662 Properties=species:S:1:pos:R:3:forces:R:3

	def RFE_to_xyz_single(self, R, F, E):
		z = self.z
		s = f'{len(z)}\n'
		s += f'{E[0]:.5e}\n'
		for i in range(0,len(R),3):
			s += f"{z[i//3]:<3}{R[i]:<13.5e}{R[i+1]:<13.5e}{R[i+2]:<13.5e}"
			s += f"{F[i]:<13.5e}{F[i+1]:<13.5e}{F[i+2]:<13.5e}\n"
		return s

	def save_command(self):
		super().save_command()
		from time import time 

		t0 = time()
		self.save_cluster_xyz()
		print(f'Took {time() - t0} seconds')