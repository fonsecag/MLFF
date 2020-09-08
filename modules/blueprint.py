from util import*
from run import MainHandler

# if NOT inheriting from another module
class CommandHandler(MainHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)

	def run_command():
		self.print_stage('Do nothing')
		self.do_nothing()
		

	def save():
		print_ongoing_progress('doing nothing')
		self.do_nothing()
		print_ongoing_progress('done nothing', True)


# if inheriting from another module
# Note: if multiple inheritance is needed to mix methods, duplicate method 
# conflicts need to be handled manually. Consider importing classes and using 
# their methods without inheritance if needed.
class CommandHandler(ParentCommandHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args)

	def run_command():
		# only if you want to run the parent process as well 
		super().run_command()

		self.print_stage('Do nothing')
		self.do_nothing()

	def save():
		# only if you want to save elements from parent process as well
		super().save()

		print_ongoing_progress('doing nothing')
		self.do_nothing()
		print_ongoing_progress('done nothing', True)
