import config as cfg

from funcs_epoching import *
from funcs_erps import *
from funcs_decoding import *


def main():

    if not os.path.isfile(cfg.eeg.epoch_file):
        get_epochs(cfg)
    else:
        print('Epochs file already exists: ' + cfg.eeg.epoch_file)

    if cfg.eeg.noise_norm == 'unn':
        get_erp(cfg)
    elif cfg.eeg.decode == 'image':
        decode_img(cfg)
    elif cfg.eeg.decode == 'object':
        decode_obj(cfg)
    elif cfg.eeg.decode == 'object-cross':
        decode_obj_cross(cfg)


if __name__ == '__main__':
    main()
