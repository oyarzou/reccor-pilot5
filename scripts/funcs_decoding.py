from project_utils import *

import numpy as np
from tqdm import tqdm

def decode_img(cfg):
    import time
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    print('TG-img for subject {}, iter: {}'.format(cfg.eeg.suj, cfg.eeg.img_iter))

    data = load_data(cfg.eeg.epoch_file)
    data = subsample_data(data, cfg.eeg.img_subsample_factor)

    dshape = data['eeg'].shape

    TG = np.full((dshape[0],dshape[0],dshape[3],dshape[3]), np.nan)
    t_init = time.time()
    for p in tqdm(range(cfg.eeg.img_nperms)):
        pstrials, binsize = get_pseudotrials(data)
        print('binsize: {}, N of pstrials: {}'.format(binsize, pstrials.shape[1]))

        ps_ixs = np.arange(pstrials.shape[1])
        train_ix = ps_ixs[:int(len(ps_ixs) * (1- cfg.eeg.img_testsize))]
        test_ix = ps_ixs[int(len(ps_ixs) * (1- cfg.eeg.img_testsize)):]

        for cA in range(dshape[0]):
        	for cB in range(cA+1,dshape[0]):
        		for t in range(dshape[3]):
        			train_x = np.array((pstrials[cA,train_ix,:,t], pstrials[cB,train_ix,:,t]))
        			train_x = np.reshape(train_x,(len(train_ix)*2, dshape[2]))

        			test_x = np.array((pstrials[cA,test_ix,:,:], pstrials[cB,test_ix,:,:]))
        			test_x = np.reshape(test_x,(len(test_ix)*2,dshape[2],dshape[3]))

        			train_y = np.array([1] * len(train_ix) + [2] * len(train_ix))
        			test_y = np.array([1] * len(test_ix) + [2] * len(test_ix))

        			classifier = LinearSVC(penalty = 'l2',
                                            loss = 'hinge',
                                            C = .5,
                                            multi_class = 'ovr',
                                            fit_intercept = True,
                                            max_iter = 10000)
        			classifier.fit(train_x, train_y)

        			for tt in range(dshape[3]):
        				pred_y = classifier.predict(test_x[:,:,tt])
        				acc_score = accuracy_score(test_y,pred_y)
        				TG[cA,cB,t,tt] += acc_score

    #TG = np.nanmean(TG, axis=0)
    TG = TG / cfg.eeg.img_nperms
    out_dict = {
				'DA': TG,
				'id': data['id'],
                'cat': data['cat'],
                'obj': data['obj'],
                'time': data['time'],
                'subject': cfg.eeg.suj,
                'n_perms': cfg.eeg.img_nperms
                }

    elapsed = time.time() - t_init
    print('time elapsed: {}'.format(str(elapsed)))

    dump_data(out_dict, cfg.eeg.DAimg_file)


def decode_obj(cfg):
    import time
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    print('TGobj for subject {}'.format(cfg.eeg.suj))

    data = load_data(cfg.eeg.epoch_file)
    data = subsample_data(data, cfg.eeg.obj_subsample_factor)

    train_ixs, test_ixs = get_sets(data['obj'],
                                    cfg.eeg.obj_testsize,
                                    cfg.eeg.obj_pstrial_binsize,
                                    cfg.eeg.obj_nperms)
    trn_len = train_ixs.shape[2]
    tst_len = test_ixs.shape[2]

    pstrial_size = int(trn_len / cfg.eeg.obj_pstrial_binsize)

    mtrials = data['eeg'].mean(axis=1)
    tshape = mtrials.shape

    objs, objs_counts = np.unique(data['obj'], return_counts=True)
    DA = np.full((cfg.eeg.obj_nperms, tst_len*len(objs), len(objs), tshape[2], tshape[2]), np.nan)

    t_init = time.time()
    for p in tqdm(range(cfg.eeg.obj_nperms)):

        tr_x = np.array([mtrials[x] for x in train_ixs[p].astype(int)])
        tr_x = np.mean(tr_x.reshape((len(objs), -1, cfg.eeg.obj_pstrial_binsize,) + tshape[1:]), axis=2)
        tr_x = np.reshape(tr_x, (len(objs) * pstrial_size, tshape[1],tshape[2]))

        train_y = np.repeat(objs,pstrial_size)

        test_ix = np.reshape(test_ixs[p],len(objs)*tst_len).astype(int)
        test_y = np.array(data['obj'])[test_ix]

        for t in range(5, tshape[2] - 5):
        	wstart = t - 5 if t - 5 > 0 else 0
        	wend = t + 5 if t + 5 < tshape[2] else tshape[2] - 1

        	train_x = tr_x[:, :, wstart:wend]
        	train_x = train_x.reshape((train_x.shape[0],-1))

        	classifier = LinearSVC(penalty = 'l2',
                                    loss = 'hinge',
                                    C = .5,
                                    multi_class = 'ovr',
                                    fit_intercept = True,
                                    max_iter = 10000)
        	classifier.fit(train_x, train_y)

        	for tt in range(5, tshape[2] - 5):
        		tt_wstart = tt - 5 if tt - 5 > 0 else 0
        		tt_wend = tt + 5 if tt + 5 < tshape[2] else tshape[2] - 1

        		test_x = mtrials[test_ix, :, tt_wstart:tt_wend]
        		test_x = test_x.reshape((test_x.shape[0],-1))

        		R = np.transpose(test_x)
        		bias = [[x] for x in classifier.intercept_]

        		sc = np.dot(classifier.coef_, R) + bias
        		sc = np.transpose(sc)
        		p_obj = np.array([np.exp(x)/sum(np.exp(x)) for x in sc]) #softmax

        		DA[p,:,:,t,tt] = p_obj

    elapsed = time.time() - t_init
    print('time elapsed: {}'.format(str(elapsed)))

    DA = np.reshape(DA,(cfg.eeg.obj_nperms * len(objs) * tst_len,)  + DA.shape[2:])
    DA_id = np.array(data['id'])[np.array(test_ixs).flatten().astype(int)]

    imDA = np.full((tshape[0],len(objs),tshape[2],tshape[2]),np.nan)
    for i in range(len(data['id'])):
    	imDA[i,:,:,:] = DA[DA_id == data['id'][i],:,:,:].mean(axis=0)

    del DA

    test_id, test_id_count = np.unique(DA_id, return_counts=True)

    out_dict = {
    			'DA': imDA,
				'id': data['id'],
    			'objs': data['obj'],
                'cat': data['cat'],
    			'time': data['time'],
    			'subject': cfg.eeg.suj,
                'test_id': test_id,
				'test_count': test_id_count
                }

    dump_data(out_dict, cfg.eeg.obj_file)


def decode_obj_cross(cfg):
    import time
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    print('TGobj-cross for subject {}, cond: {}-{}'.format(cfg.eeg.suj,
                                                        cfg.eeg.obj_train,
                                                        cfg.eeg.obj_test))

    data = load_data(cfg.eeg.epoch_file)
    data = subsample_data(data, cfg.eeg.obj_subsample_factor)

    data_train = subset_data(data, 0, np.array(data['cat']) == cfg.eeg.obj_train)
    data_test = subset_data(data, 0, np.array(data['cat']) == cfg.eeg.obj_test)

    if sum(data_test['obj'] == data_train['obj']) == len(data_test['obj']):
        objs, objs_counts = np.unique(data_train['obj'], return_counts=True)

        set_len = np.min(objs_counts)
        trn_len = int(np.round(set_len * (1 - cfg.eeg.obj_testsize)))
        tst_len = int(np.round(set_len * (cfg.eeg.obj_testsize)))
        pstrial_size = int(trn_len / cfg.eeg.obj_pstrial_binsize)

        train_ixs = np.full((cfg.eeg.obj_nperms, len(objs),trn_len), np.nan)
        test_ixs = np.full((cfg.eeg.obj_nperms, len(objs),tst_len), np.nan)
        for i in range(cfg.eeg.obj_nperms):
        	for o in range(len(objs)):
        		o_ix = np.where(np.array(data_train['obj']) == objs[o])[0]
        		np.random.seed(i)
        		o_ix = np.random.permutation(o_ix)

        		train_ixs[i,o,:] = o_ix[:trn_len]
        		test_ixs[i,o,:] = o_ix[trn_len:set_len]

    else:
        print('Data is not balanced across categories. It is not possible to use the algorithm for the construction of train/test sets')
        return

    mtrials_train = data_train['eeg'].mean(axis=1)
    mtrials_test = data_test['eeg'].mean(axis=1)
    tshape = mtrials_train.shape

    DA = np.full((cfg.eeg.obj_nperms, tst_len*len(objs), len(objs),tshape[2],tshape[2]), np.nan)

    t_init = time.time()
    for p in tqdm(range(cfg.eeg.obj_nperms)):

        tr_x = np.array([mtrials_train[x] for x in train_ixs[p].astype(int)])
        tr_x = np.mean(tr_x.reshape((len(objs), -1, cfg.eeg.obj_pstrial_binsize,) + tshape[1:]), axis=2)
        tr_x = np.reshape(tr_x, (len(objs) * pstrial_size, tshape[1],tshape[2]))

        train_y = np.repeat(objs,pstrial_size)

        test_ix = np.reshape(test_ixs[p],len(objs)*tst_len).astype(int)
        test_y = np.array(data_test['obj'])[test_ix]

        for t in range(5, tshape[2] - 5):
        	wstart = t - 5 if t - 5 > 0 else 0
        	wend = t + 5 if t + 5 < tshape[2] else tshape[2] - 1

        	train_x = tr_x[:, :, wstart:wend]
        	train_x = train_x.reshape((train_x.shape[0],-1))

        	classifier = LinearSVC(penalty = 'l2',
                                    loss = 'hinge',
                                    C = .5,
                                    multi_class = 'ovr',
                                    fit_intercept = True,
                                    max_iter = 10000)
        	classifier.fit(train_x, train_y)

        	for tt in range(5, tshape[2] - 5):
        		tt_wstart = tt - 5 if tt - 5 > 0 else 0
        		tt_wend = tt + 5 if tt + 5 < tshape[2] else tshape[2] - 1

        		test_x = mtrials_test[test_ix, :, tt_wstart:tt_wend]
        		test_x = test_x.reshape((test_x.shape[0],-1))

        		R = np.transpose(test_x)
        		bias = [[x] for x in classifier.intercept_]

        		sc = np.dot(classifier.coef_, R) + bias
        		sc = np.transpose(sc)
        		p_obj = np.array([np.exp(x)/sum(np.exp(x)) for x in sc]) #softmax

        		DA[p,:,:,t,tt] = p_obj

    elapsed = time.time() - t_init
    print('time elapsed: {}'.format(str(elapsed)))

    DA = np.reshape(DA,(cfg.eeg.obj_nperms * len(objs) * tst_len,)  + DA.shape[2:])
    DA_id = np.array(data_test['id'])[np.array(test_ixs).flatten().astype(int)]

    imDA = np.full((tshape[0],len(objs),tshape[2],tshape[2]),np.nan)
    for i in range(len(data_test['id'])):
    	imDA[i,:,:,:] = DA[DA_id == data_test['id'][i],:,:,:].mean(axis=0)

    del DA

    test_id, test_id_count = np.unique(DA_id, return_counts=True)

    out_dict = {
    			'DA': imDA,
				'id': data_test['id'],
    			'objs': objs,
                'cat': cfg.eeg.obj_test,
    			'time': data['time'],
    			'subject': cfg.eeg.suj,
                'test_id': test_id,
				'test_count': test_id_count
                }

    dump_data(out_dict, cfg.eeg.DAoc_file)


def subsample_data(dat, sub_factor):
	dshape = dat['eeg'].shape
	sub_ix = list(range(0,dshape[3],sub_factor))
	dat['eeg'] = dat['eeg'][:,:,:,sub_ix]
	dat['time'] = dat['time'][sub_ix]
	dat['id'] = [int(x) for x in dat['id']]

	return(dat)


def get_pseudotrials(dat):
	import numpy as np

	shape = dat['eeg'].shape
	k = shape[1]
	l = np.int(shape[1] / k)

	while l < 5:
		k = k - 1
		l = np.int(shape[1] / k)

	dat['eeg'] = dat['eeg'][:,np.random.permutation(shape[1]),:,:]
	dat['eeg'] = dat['eeg'][:,:l*k,:,:]

	pst = np.reshape(dat['eeg'], (shape[0], k, l, shape[2],shape[3]))
	pst = pst.mean(axis=1)

	return(pst, k)


def subset_data(dat, dim, mask):
	import numpy as np

	d = {}
	for i in dat.keys():
		if not i in ['trial', 'chans', 'time']:
			idxs = [slice(None)] * len(np.array(dat[i]).shape)
			idxs[dim] = mask
			dd = np.copy(dat[i])
			d[i] = dd[tuple(idxs)]

	return(d)


def get_sets(objects, testsize, binsize, nperms):
	import numpy as np

	objs, objs_counts = np.unique(objects, return_counts=True)

	set_len = np.min(objs_counts)
	trn_len = int(np.round(set_len * (1 - testsize)))
	tst_len = set_len - trn_len

	train_ixs = np.full((nperms, len(objs), trn_len), np.nan)
	test_ixs = np.full((nperms, len(objs), tst_len), np.nan)

	for i in range(nperms):
		for o in range(len(objs)):
			o_ix = np.where(np.array(objects) == objs[o])[0]
			np.random.seed(i)
			o_ix = np.random.permutation(o_ix)

			train_ixs[i,o,:] = o_ix[:trn_len]
			test_ixs[i,o,:] = o_ix[trn_len:set_len]

	return(train_ixs, test_ixs)
