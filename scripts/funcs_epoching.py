from project_utils import *

import glob
import mne
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_epochs(cfg):

    if cfg.eeg.suj is not None:
        print('processing epochs for subject ' + cfg.eeg.suj)

        dat = extract_epochs(cfg)
        dshape = dat['eeg'].shape

        if cfg.eeg.noise_norm == 'unn':
            print("\n>>> Univariate noise normalization  <<<")
            dat['eeg'] = np.array([[[unn(dat['eeg'][x, y, z, :], dat['time'])
                                    for z in range(dshape[2])]
                                    for y in range(dshape[1])]
                                  for x in tqdm(range(dshape[0]))])

        elif cfg.eeg.noise_norm[:3] == 'mnn':
            print("\n>>> Multivariate noise normalization  <<<")
            dat['eeg'] = mnn(dat['eeg'], cfg.eeg.noise_norm[4])

        elif cfg.eeg.noise_norm == 'raw':
            print("\n>>> No noise normalization performed <<<")

        dump_data(dat, cfg.eeg.epoch_file)


def extract_epochs(cfg):

    files = glob.glob('{}/rec*.eeg'.format(cfg.eeg.dirs['eeg_dir']))
    all_sessions = [x.split('rec')[1].split('.')[0] for x in files]

    stimlist = pd.read_csv(cfg.eeg.dirs['stim_file'])
    stimlist = stimlist.drop_duplicates(subset=['id'])

    sessions = [cfg.eeg.suj, str(100 + int(cfg.eeg.suj))]

    for ses in sessions:

        if ses not in all_sessions:
            print("Session " + ses + " not found!!!")
            continue

        else:
            # Get eeg data
            eeg_filename = 'rec' + ses + '.vhdr'
            eegfile = os.path.join(cfg.eeg.dirs['eeg_dir'], eeg_filename)
            raw = mne.io.read_raw_brainvision(eegfile, preload=True)

            # Get events from eeg
            events, events_id = mne.events_from_annotations(raw)
            events = events[1:, :]

            # Get behavioral data
            logfile = os.path.join(cfg.eeg.dirs['beh_dir'], 'img_{}_parsed.csv'.format(ses))
            logfiles = glob.glob('{}/*.csv'.format(cfg.eeg.dirs['beh_dir']))
            if not logfile in logfiles:
                behfile = os.path.join(cfg.eeg.dirs['beh_dir'], 'img_{}.txt'.format(ses))
                get_csv_and_parse_rows(behfile, cfg.eeg.logfile_header)

            imdat = pd.read_csv(logfile, na_values='NaN')
            imdat['ncat'], cat_codes = pd.factorize(imdat.cat, sort=True)

            ievents_, img_ix_, seq_ix_, seqs = check_imgs_events(events, imdat)

            eeg_picks = mne.pick_types(raw.info, eeg=True)

            raw = raw.copy().filter(l_freq=cfg.eeg.highpass,
                                    h_freq=cfg.eeg.lowpass,
                                    picks=eeg_picks)

            # preload argument force loading of data into memory
            epochs = mne.Epochs(raw, ievents_, tmin=cfg.eeg.trialwin[0],
                                tmax=cfg.eeg.trialwin[1], picks='eeg',
                                baseline=(None, 0), preload=True,
                                reject=None)


            eeg_ = epochs.get_data()
            time_ = epochs.times
            ids_ = ievents_[:, 2]


            try:
                eeg
            except NameError:
                eeg = eeg_
                img_dat = imdat
                channels = epochs.ch_names
                time = time_
                ids = ids_
                ievents = ievents_
                img_ix = img_ix_
                seq_ix = seq_ix_
            else:
                print("\n>>> Concatenating epochs  <<<")
                eeg = np.concatenate((eeg, eeg_), axis=0)
                ids = np.concatenate((ids, ids_), axis=0)
                ievents = np.concatenate((ievents, ievents_), axis=0)
                img_ix = np.concatenate((img_ix, img_ix_), axis=0)
                seq_ix = np.concatenate((seq_ix, seq_ix_), axis=0)
                img_dat = img_dat.append(imdat)
                del eeg_, ievents_, img_ix_, seq_ix_, imdat

            del epochs, raw

    dat = sort_data_im(eeg, ids, channels, time, img_ix, seq_ix)
    print('epochs shape: ' + str(dat['eeg'].shape))

    return(dat)

def get_csv_and_parse_rows(filename, header):
    ncol = len(header)
    res = []
    with open(filename) as infile:             #Read CSV
    #    header = img_names      #Get Header
        for line in infile:                    #Iterate each line
            val = line.split(",")
            if len(val) == ncol:                  #Check if ncol elements in each line
                res.append(val)
            else:
                res.extend( [val[i:i+ncol] for i in range(0, len(val), ncol)] )     #Else split it.

    df = pd.DataFrame(res, columns=header)
    outname = filename[:-4] + '_parsed.csv'
    df.to_csv(outname)


def check_imgs_events(evs, dat, get_dist=False):

    dat.reset_index(inplace=True)
    dat['obj'] = pd.to_numeric(dat['obj'])
    eix = 0

    img_evs = np.full((0, 3), 0)
    dat_ix = []
    seq_ix = []
    seq = []
    print("\n>>> Checking trigger events <<<")

    reix = 0
    revs = evs.copy()
    for i in tqdm(range(len(dat))):

        if get_dist is False and np.isnan(dat.obj.iloc[i]):
            continue

        ta = dat.triga.iloc[i]
        tb = dat.trigb.iloc[i]
        tc = dat.trigc.iloc[i]
        td = dat.trigd.iloc[i]

        control = True
        while(control):
            revs = revs[reix:]
            reix = np.where(revs[:, 2] == ta)[0][0]
            if (revs[reix + 1, 2] == tb
                    and revs[reix + 2, 2] == tc
                    and revs[reix + 3, 2] == td):
                eix = np.where(evs[:, 0] == revs[reix, 0])[0][0]
                control = False
            reix = reix + 1

        obj, img_id = get_imgid(dat.iloc[i])
        e = [evs[eix, 0], 0, img_id]

        img_evs = np.append(img_evs, [np.array(e).astype(int)], axis=0)
        dat_ix = np.append(dat_ix, i)
        seq_ix = np.append(seq_ix, dat.i.iloc[i])
        seq = np.append(seq, dat.seq.iloc[i])
    if len(img_evs) !=  sum(~np.isnan(dat.obj.values)):
        print('Something weird happened: number of trials does not match')

    return img_evs, dat_ix, seq_ix, seq


def get_imgid(d):

    if np.isnan(d.obj):
        o = 100
        i = 100
    else:
        o = int(d.obj)
        i = ((int(d.ncat) + 1) * 10000) + (int(d.obj) * 1000) + int(d.img)

    return(o, i)


def sort_data_im(eeg, dat_id, chans, time, im_ix, sq_ix):
    dshape = eeg.shape
    ids, id_counts = np.unique(dat_id, return_counts=True)

    sdat = np.empty((len(ids), id_counts.min(), dshape[1], dshape[2]))
    nids = np.zeros((len(ids)))
    nimix = np.empty((len(ids), id_counts.min()))
    nsqix = np.empty((len(ids), id_counts.min()))
    for c in tqdm(range(len(ids))):
        d = eeg[dat_id == ids[c]]
        im_ = im_ix[dat_id == ids[c]]
        sq_ = sq_ix[dat_id == ids[c]]

        shuffle_ix = np.arange(id_counts.min())
        np.random.shuffle(shuffle_ix)
        selected_trials = shuffle_ix[:id_counts.min()]
        sdat[c] = d[selected_trials, :, :]
        nids[c] = ids[c]
        nimix[c] = im_[selected_trials]
        nsqix[c] = sq_[selected_trials]

    ndat = {
            'eeg': sdat,
            'img_ix': nimix,
            'seq_ix': nsqix,
            'id': nids,
            'cat': [get_cat(int(x)) for x in nids],
            'obj': [get_obj(int(x)) for x in nids],
            'trial': np.arange(id_counts.min()),
            'chans': chans,
            'time': time
            }

    return(ndat)


def unn(dat, t):
    bl = t <= 0

    bl_mean = np.mean(dat[bl])
    bl_sd = np.std(dat[bl])

    zdat = (dat - bl_mean) / bl_sd

    return(zdat)


def mnn(dat, mnn_dim):
    from sklearn.discriminant_analysis import _cov
    import scipy

    # organize data by Conditions
    sigma_ = np.empty((dat.shape[0], dat.shape[2], dat.shape[2]))
    for c in range(dat.shape[0]):
        # subset trials of condition c.
        d = dat[c]

        # If computing covariace matrices for each time point
        if mnn_dim == 't':
            # Computing sigma for each time point, then averaging across time
            sigma_[c, :, :] = np.mean([_cov(d[:, :, t],
                                      shrinkage='auto')
                                      for t in range(d.shape[2])], axis=0)

        # If computing covariace matrices for each epoch (repetition)
        elif mnn_dim == 'e':
            # Computing sigma for each epoch, then averaging across epochs
            sigma_[c, :, :] = np.mean([_cov(np.transpose(d[e, :, :]),
                                       shrinkage='auto')
                                       for e in range(d.shape[0])], axis=0)

    sigma = sigma_.mean(axis=0)	 # Averaging sigma across conditions
    # Computing the inverse of sigma
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
    dshape = dat.shape
    rs_dat = np.reshape(dat, (-1, dshape[2], dshape[3]))
    whitened_dat = (rs_dat.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
    whitened_dat = np.reshape(whitened_dat, dshape)

    return(whitened_dat)
