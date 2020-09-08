import sys
from .clustererror import ClusterErrorHandler
from util import*

class PlotClusterErrorHandler(ClusterErrorHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)

		self.n_stages = self.n_main_stages + self.n_substages


	n_substages = 2
	# loading info.p, plotting

	def run_command(self):
		self.print_stage('Loading info file')
		self.load_info_file(self.args['info_file'])

		self.print_stage('Plotting')
		self.plot_cluster_error()
		
	def save_command(self):
		pass
		# super().save_command()
		# No need, plots do their own thing
