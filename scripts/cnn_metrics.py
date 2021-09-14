from project_utils import *
import config as cfg

def get_metrics(dat):
    import scipy.stats
    from tqdm import tqdm

    dshape = dat['da'].shape
    dat['id'] = dat['id'].flatten()
    dat['img'] = dat['img'].flatten()
    dat['obj'] = dat['obj'].flatten()

    objs = np.unique(dat['obj'])

    target_mask = np.array([objs == x for x in dat['obj']])

    p_target = dat['da'][target_mask]
    p_target = p_target[:,np.newaxis]

    p_distract = dat['da'][~target_mask]
    p_distract = np.reshape(p_distract, (dshape[0],-1,))

    denominator = (p_distract + p_target) ** -1

    hr = p_target * denominator
    hr = np.nanmean(hr, axis=1)

    fa = np.full(hr.shape, np.nan)
    for i in tqdm(range(dshape[0])):
        fa_mask = np.array(dat['obj']) != dat['obj'][i]
        fa_o = 1 - np.nanmean(hr[fa_mask], axis=0)
        fa[~fa_mask] = fa_o

    i1 = scipy.stats.norm.ppf(hr) - scipy.stats.norm.ppf(fa)
    i1 = [5 if x > 5 else -5 if x < -5 else x for x in i1]

    metrics = {
                'id': dat['id'],
                'hr': hr,
                'fa': fa,
                'i1': i1
    }

    return(metrics)

def main():
    data_svm = load_data(cfg.cnn.da_svm_file)
    metrics_svm = get_metrics_svm(data_svm)
    dump_data(metrics_svm, cfg.cnn.metrics_svm_file)

    data_ft = load_data(cfg.cnn.da_ft_file)
#    obj, img = np.array([[x for x in str.split(i[:-4],'/')[-2:]] for i in data_ft['file']]).T
    data_ft['id'] = np.array(['{}_{}'.format(data_ft['obj'][x], data_ft['img'][x]) for x in range(len(data_ft['obj']))])

    metrics_ft = get_metrics(data_ft)


if __name__ == '__main__':
    main()
