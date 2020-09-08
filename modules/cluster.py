import sys

from run import MainHandler
from util import*

class ClusterHandler(MainHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)
		self.n_stages = self.n_main_stages + self.n_substages

	n_substages = 1
	def run_command(self):
		self.print_stage('Cluster data')
		self.cluster_data()

	def cluster_data(self):
		from funcs.cluster import cluster_do

		cluster_path=self.args['cluster_file']
		if cluster_path is None:
			scheme=self.call_para('clusters','init_scheme')
			self.cluster_indices=cluster_do(self,scheme)
			cl_ind = self.cluster_indices

		else:
			if not os.path.exists(cluster_path):
				print_error(f"No file found under {cluster_path}.")

			print_ongoing_process(f"Loading clusters from {cluster_path}")
			cl_ind=np.load(cluster_path,allow_pickle=True)
			var_index=self.para['clusters'][0]['var_index']
			N=len(self.vars[var_index])
			print_ongoing_process(f"Loaded clusters from {cluster_path}",True)

			if N!=len(np.concatenate(cl_ind)):
				print_warning(
					f"Given cluster file might contain different amount of points than given dataset. Length of vars as given by var_index {var_index}: {N}. Number of indices in given cluster file: {len(np.concatenate(cl_ind))}.")

			self.cluster_indices=cl_ind

		self.info['cluster_indices'] = self.cluster_indices

		# SUMMARY
		self.print_cluster_summary(cl_ind)

	def print_cluster_summary(self, cl_ind):
		summary_table={}
		summary_table['clusters']=len(cl_ind)
		l=[len(x) for x in cl_ind]
		summary_table['size avg']=int(np.average(l))
		summary_table['size min']=np.min(l)
		summary_table['size max']=np.max(l)
		print_table("Clusters summary:",None,None,summary_table)

	def save_command(self):

		if self.call_para('clusters','save_npy'):
			cluster_path = os.path.join(self.storage_dir, 'cluster_indices.npy')
			np.save(cluster_path, self.cluster_indices)

	
def prepare_classifier(self):

	print_ongoing_process("Preparing classifier")
	cl_ind=self.cluster_indices
	n_cl=len(cl_ind)

	R=self.vars[self.call_para('classification','var_index')]
	X=np.concatenate([R[x] for x in cl_ind])
	Y=np.zeros(len(X))

	tot,prev=0,0
	for i in range(n_cl):
		cl_len=len(cl_ind[i])
		tot=prev+cl_len
		Y[prev:tot]=i
		prev=tot

	from sklearn import preprocessing
	lb = preprocessing.LabelBinarizer()
	Y = lb.fit_transform(Y)

	print_ongoing_process("Preparing classifier",True)

	scaler=None
	if self.get_para('classification','scaler_func') is not None:
		print_ongoing_process("Preparing scaler")
		scaler=self.call_para('classification','scaler_func',
			args=[self,X]
		)
		self.classification_x_scaler=scaler
		X=scaler.transform(X)
		print_ongoing_process("Preparing scaler",True)
	else:
		self.scaler=None
		print_warning("No scaler chosen for classification.")


	N,X_sub,Y_sub,X_rest,Y_rest=len(Y),None,None,None,None
	n_points=self.call_para('classification','n_points')
	print_ongoing_process("Preparing train/test set")

	if n_points>1:
		n_points=int(n_points)
		sub_ind=np.random.choice(N,n_points,replace=False)
		X_sub,Y_sub=X[sub_ind],Y[sub_ind]
		sub_rest=np.delete(np.arange(N),sub_ind)
		X_rest,Y_rest=X[sub_rest],Y[sub_rest]

	elif n_points==1:
		X_sub=X
		Y_sub=Y

	elif n_points<1:
		n_points=int(n_points*N)
		sub_ind=np.random.choice(N,n_points,replace=False)
		X_sub,Y_sub=X[sub_ind],Y[sub_ind]
		sub_rest=np.delete(np.arange(N),sub_ind)
		X_rest,Y_rest=X[sub_rest],Y[sub_rest]

	print_ongoing_process("Preparing train/test set",True)

	print_ongoing_process("Training classifier")
	classifier=self.call_para('classification','class_func',
		args=[self,X_sub,Y_sub]
	)
	print_ongoing_process("Training classifier",True)

	#quick check
	# tot,prev=0,0
	# for x in cl_ind:
	#     tot=tot+len(x)
	#     print_debug(f"From: {prev} to {tot}:")
	#     print(Y[prev:tot])
	#     prev=tot
	#print(len(R),len(np.concatenate(cl_ind)),len(Y))


	if self.call_para('classification','perform_test') and n_points!=1:
		summary=self.call_para('classification','test_func',
			args=[self,classifier,X_rest,Y_rest,X_sub,Y_sub]
		)
	else:
		summary={}

	#PRINT SUMMARY
	summary['N total']=N
	summary['N train']=len(X_sub)
	summary['N test']=len(X_rest)

	print_table('Classifier summary',None,None,summary,width=15)

	self.classifier=classifier

def save_classifier(self):
	path=self.storage_dir
	print_ongoing_process('Saving classifer x scaler')
	if getattr(self,'classification_x_scaler',None) is not None:
		joblib.dump(self.classification_x_scaler,os.path.join(path,"clf_x_scaler.z"))
	print_ongoing_process(f'Saved classifer x scaler {os.path.join(path,"clf_x_scaler.z")}',True)

	print_ongoing_process('Saving classifer')
	if getattr(self,'classifier',None) is not None:
		joblib.dump(self.classifier,os.path.join(path,"clf.z"))
	print_ongoing_process(f'Saved classifer {os.path.join(path,"clf.z")}',True)
