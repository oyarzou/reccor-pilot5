from project_utils import *
import config as cfg

import glob
import pandas as pd
import numpy as np
from scipy.stats import norm
import itertools

def get_behavioral_data(files, first_run = False):
    db = pd.DataFrame()
    for i in range(len(files)):
        d = pd.read_csv(files[i])
        d = d[d.left.notnull()]

        if first_run == True:
            d = d[d['trials.thisRepN'] == 0]

        d['ses'] = i
        d['sesfile'] = files[i]

        stim_id = [x[x.find('obj')+3:x.find('.png')] for x in d.stim]
        d['stim_id'] = stim_id
        d['obj_id'] = [x.split('_')[0] for x in stim_id]
        d['pic_id'] = [x.split('_')[1] for x in stim_id]

        left_id = [x[x.find('im')+2:x.find('.png')] for x in d.left]
        d['left_id'] = left_id
        right_id = [x[x.find('im')+2:x.find('.png')] for x in d.right]
        d['right_id'] = right_id

        d['resp'] = [x[0] for x in d['key_resp.keys']]
        d['correct'] = d.resp == d.ans
        d['ncorrect'] = [int(x) for x in d.correct]

        d['rt'] = d['key_resp.rt']

        db = db.append(d)

    db.reset_index(inplace=True)
    db2 = db[['participant','stim_id','obj','obj_id','pic_id',
                'left_id','right_id','ans','resp','key_resp.keys',
                'correct','ncorrect','rt']]

    return(db2)

def get_behavioral_accuracy(data):
    hr = data.groupby(['stim_id','obj_id','pic_id']).mean()
    hr.reset_index(inplace = True)
    hr['obj_id'] = [int(x) for x in hr.obj_id]
    hr['pic_id'] = [int(x) for x in hr.pic_id]
    hr = hr.sort_values(by=['obj_id','pic_id'])

    rt = []
    hri = []
    fai = []
    im_lab = []
    obj = []
    for i in hr.stim_id.unique():
        hr0 = float(hr[hr.stim_id == i].correct)
        rt0 = float(hr[hr.stim_id == i].rt)
        o = hr[hr.stim_id == i].obj_id.iloc[0]
        hrj = hr[hr.obj_id != o].correct.mean()
        fa0 = 1 - hrj

        obj.append(o)
        hri.append(hr0)
        rt.append(rt0)
        fai.append(fa0)
        im_lab.append(i)

    i1 = norm.ppf(hri) - norm.ppf(fai)
    i1 = [5 if x > 5 else -5 if x < -5 else x for x in i1]

    dat = pd.DataFrame({
                        'obj': obj,
                        'im_lab': im_lab,
                        'hri': hri,
                        'fai': fai,
                        'i1': i1,
                        'rt': rt})
    return(dat)

def main():

    data_files = glob.glob(cfg.beh.data_dir + "*.csv")

    data = get_behavioral_data(data_files, first_run = False)
    accuracy = get_behavioral_accuracy(data)

    data_suj = {'s' + str(int(x)): data[data.participant == x] for x in data.participant.unique()}
    acc_suj = {x: get_behavioral_accuracy(data_suj[x]) for x in data_suj.keys()}

    for i in acc_suj.keys():
        accuracy['hr_' + i] = acc_suj[i].hri
        accuracy['fa_' + i] = acc_suj[i].fai
        accuracy['i1_' + i] = acc_suj[i].i1
        accuracy['rt_' + i] = acc_suj[i].rt

    accuracy.to_csv(cfg.beh.metrics_file)



if __name__ == '__main__':
    main()
