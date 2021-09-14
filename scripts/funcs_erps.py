from project_utils import *

def get_erp(cfg):
    data = load_data(cfg.eeg.epoch_file)

    for r in range(len(cfg.eeg.rois.keys())):
        roi_mask = np.isin(np.array(data['chans']), cfg.eeg.rois[list(cfg.eeg.rois.keys())[r]])
        eeg_roi = data['eeg'][:,:,roi_mask,:]

        erp = eeg_roi[data['seq_ix'] == 1].mean(axis=(0,1))

        try:
            erps
        except NameError:
            erps = np.empty((len(list(cfg.eeg.rois.keys())),len(data['time'])))

        erps[r,:] = erp

    out_dict = {
                'erps': erps,
                'rois': cfg.eeg.rois,
                'time': data['time']
                }

    dump_data(out_dict, cfg.eeg.erp_file)
