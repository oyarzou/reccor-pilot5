import numpy as np

n_array = 22

#combinations = np.full((2,2,3,5,2,2), np.nan)
combinations = np.full((n_array), np.nan).astype(str)

ix = 0
for s in ['0','1']:
    for n in ['u','m']:
        if n == 'm':
            for d in ['i','o','oc']:
                if d == 'i':
                    for it in ['0','1','2','3','4']:
                        combinations[ix] = '{}_{}_{}_{}_{}-{}'.format(s,n,d,it,'n','n')
                        ix += 1

                elif d == 'oc':
                    for te in ['0','1']:
                        for tr in ['0','1']:
                            combinations[ix] = '{}_{}_{}_{}_{}-{}'.format(s,n,d,'n',te,tr)
                            ix += 1

                elif d == 'o':
                    combinations[ix] = '{}_{}_{}_{}_{}-{}'.format(s,n,d,'n','n', 'n')
                    ix += 1

                (5+4+1+1)*2


combinations
