from project_utils import *

import sys
import os

dummy = False

#Parameter space
proj_dir = get_projectDir()
mfile = 'stimlist_inhouse-metrics.csv'
dirs = {
        'beh_dir': os.path.join(proj_dir, 'matlab-paradigm', 'data'),
        'eeg_dir': os.path.join(proj_dir, 'data'),
        'stim_file': os.path.join(proj_dir, 'matlab-paradigm', 'tables', mfile),
        'out_dir': os.path.join(proj_dir, 'output_files')
        }

subjects = ['12', '13']
nns = ['unn','mnn-e']
decoding_targets = ['image', 'object', 'object-cross']
conditions = {'challenge':'1', 'control':'2'}

logfile_header = ['block', 'seq', 'i', 'ix', 'id', 'obj', 'img', 'file',
                 'd_km', 'd_kn', 'd_kh', 'd_cnn', 'd_beh',
                 'cat_km', 'cat_kh', 'cat_kb', 'cat',
                 'time', 'triga', 'trigb', 'trigc', 'trigd', 'tout']
###############################################################################

if dummy:

    suj = subjects[0]
    noise_norm = nns[0]
    decode = decoding_targets[0]
    img_iter = '0'
    obj_train = conditions['challenge']
    obj_test = conditions['control']

else:

    suj = subjects[int(sys.argv[1])]
    noise_norm = nns[int(sys.argv[2])]
    decode = decoding_targets[int(sys.argv[3])]
    img_iter = sys.argv[4] #relevant only for image decoding
    obj_train = conditions[sys.argv[5]] #relevant only for object cross decoding
    obj_test = conditions[sys.argv[6]] #relevant only for object cross decoding


print('subject: ', suj)
print('noise normalization: ', noise_norm)
print('decoding target: ', decode)
print('ing decoding, n iteration: ', img_iter)
print('obj-cross, train condition: ', obj_train)
print('obj-cross, test condition: ', obj_test)

################################################################################
#epoch config
trialwin = [-.05, .3]
bl = 'trial'  # possible values: 'trial' and 'sequence'
ep_label = 'eeg-epochs'
epoch_file = os.path.join(dirs['out_dir'],'{}_{}_s{}.pkl'.format(ep_label,
                                                    noise_norm,
                                                    suj))

highpass = .5
lowpass = 40

################################################################################
#ERP config
rois = {
    'occ': ['O1','Oz','O2'],
    'cen': ['C1','Cz','C2'],
    'fro': ['AF3','AFz','AF4']
    }

erp_file = os.path.join(dirs['out_dir'], 'eeg-ERP_s{}.pkl'.format(suj))



################################################################################
#Image decoding config
if dummy:
    img_subsample_factor = 100
    img_nperms = 2
else:
    img_subsample_factor = 5
    img_nperms = 100

img_testsize = .25
img_outlabel = 'eeg_TG-img'
DAimg_file = os.path.join(dirs['out_dir'], '{}_s{}_{}.pkl'.format(img_outlabel,
                                                        suj,
                                                        img_iter))


################################################################################
#Object decoding config
if dummy:
    obj_subsample_factor = 100
    obj_nperms = 2
else:
    obj_subsample_factor = 2
    obj_nperms = 1000

obj_pstrial_binsize = 6
obj_testsize = .25
obj_outlabel = 'eeg_TG-ova'

obj_file = os.path.join(dirs['out_dir'], '{}_s{}.pkl'.format(obj_outlabel, suj))


################################################################################
#Object cross-decoding config
oc_outlabel = 'eeg_TGcross-ova'
DAoc_file = dirs['out_dir'] + '{}_{}-{}_s{}.pkl'.format(oc_outlabel,
                                                        obj_train,
                                                        obj_test,
                                                        suj)
