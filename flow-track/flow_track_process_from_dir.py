import argparse
from utils.flow_track_utils import *


def read_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('input_dir')

	args = parser.parse_args()

	return args 

a = read_arguments()

print('Input folder directory: {}'.format(a.input_dir))


# build human detector

# build model

# run flow_track_from_dir
