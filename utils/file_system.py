import os

def check_folder_exists(path, create_if_not=True):
	if os.path.isdir(path):
		return True
	elif create_if_not:
		os.mkdir(path)
		return True
	else:
		return False
