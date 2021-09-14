from project_utils import *

#import sys

################################################################################
#Parameter space
project_dir = get_projectDir()

data_dir = '/Users/pablo/Documents/phd/RecCOR/pilot/data/behavioral_data/'
out_dir = project_dir + "/output_files/"

subjects = ['s1', 's2', 's3', 's4']


metrics_label = 'behavioral_metrics'
metrics_file = out_dir + metrics_label + '.csv'
