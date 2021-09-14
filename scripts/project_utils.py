import numpy as np
import pickle
import os

def get_projectDir():

    working_dir = os.getcwd()
    project_dir = os.path.abspath(os.path.join(working_dir, os.pardir))

    return(project_dir)

def get_obj(x):
	return str(x)[-4]

def get_im(x):
	return str(int(str(x)[-3:]))

def get_cat(x):
	return str(x)[-5]

def get_infoFromLabels(labs):
    s = np.array([x[0] for x in labs])
    bl = np.array([x[1] for x in labs])
    nn = np.array([x[2] for x in labs])

    return(s,bl,nn)

def load_data(file):
    print('loading file: ' + file)
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return(data)

def dump_data(data, filename):
    print('writing file: ' + filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def get_stimid(d):

    if np.isnan(d.obj):
        o = 100
        i = 100
    else:
        o = int(d.obj)
        i = ((int(d.ncat) + 1) * 10000) + (int(d.obj) * 1000) + int(d.img)

    return(i)


def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def get_h5(file):
    import h5py
    import pandas as pd

    ds = h5py.File(file, 'r')
    ds_dict = {x: np.squeeze(np.array(ds[x])) for x in list(ds.keys())}
    ds.close()

    if 'images' in ds_dict.keys():
        del ds_dict['images']

    df = pd.DataFrame(ds_dict)

    return(df)
#    return(ds_dict)
