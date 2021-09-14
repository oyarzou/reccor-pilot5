from project_utils import *

import os
#import sys

################################################################################
#Parameter space
project_dir = get_projectDir()

out_dir = os.path.join(project_dir, 'output_files')
stim_dir = os.path.join(project_dir, 'matlab-paradigm','stimuli','all')

layer = ['classifier', '4']
layer_label = '_'.join(layer)

features_file = os.path.join(out_dir, 'features_alexnet_{}.pkl'.format(layer_label))
pca_file = os.path.join(out_dir, 'featuresPCA_alexnet_{}.pkl'.format(layer_label))

da_svm_file = os.path.join(out_dir, 'DAobj_alexnet_{}.pkl'.format(layer_label))
metrics_svm_file = os.path.join(out_dir, 'metrics_alexnet_{}.pkl'.format(layer_label))

da_ft_file = os.path.join(out_dir, 'DAobj_alexnet_ft_kar.pkl')
metrics_ft_file = os.path.join(out_dir, 'metrics_alexnet_ft.pkl')
