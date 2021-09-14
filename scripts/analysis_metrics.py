from project_utils import *
import config as cfg

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt


def plot_mean_obs(x,y,hue):
    legend = np.sort(hue.unique())
    sns.set_theme(style="whitegrid")
    # Initialize the figure
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    sns.stripplot(x=x, y=y, hue=hue, dodge=True, alpha=.25, zorder=1, hue_order=legend)

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    sns.pointplot(x=x, y=y, hue=hue, dodge=.8 - .8 / 3,
                  join=False, palette="dark", hue_order=legend,
                  markers="d", scale=.75, ci=None)

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(legend):], labels[len(legend):],
              handletextpad=0, columnspacing=1,
              loc="lower right", ncol=len(legend), frameon=True)


def plots_beh_metrics(metrics):
    metrics_names = ['i1','fai','hri','rt']
    sujs_vars = ['i1','fa','hr','rt']
    suj_columns = [not x in metrics_names for x in metrics.columns.values]

    metrics_s = metrics[metrics.columns[suj_columns]]

    metrics_sl = pd.wide_to_long(metrics_s,
                                stubnames=sujs_vars,
                                i="im_lab",
                                j="suj",
                                sep="_",
                                suffix='\\w+')
    metrics_sl.reset_index(inplace=True)

    for var in sujs_vars:
        plot_mean_obs(x=metrics_sl.obj, y=metrics_sl[var], hue=metrics_sl.suj)

    plt.plot(metrics.i1)


def plots_cnn_metrics(metrics):
    plt.plot(metrics.i1)


def compare(x,y,labs):
    import scipy.stats

    labs = ["d' of " + x for x in labs]
    comparison = pd.DataFrame({'x': x,'y': y})
    #correlation
    sns.set(style="whitegrid")
    sns.despine(bottom=True, left=True)
    s = sns.jointplot(x='x', y='y', data=comparison,
                      kind="reg", truncate=False,
                      xlim=(0, 6), ylim=(0, 6),
                      color="g", height=9,
                      marker='2')
    #plt.title("images d' for both participants")
    s.set_axis_labels(labs[0],labs[1])

    print(scipy.stats.spearmanr(comparison.x,comparison.y))
    print('mean x:' + str(np.mean(comparison.x)))
    print('mean y:' + str(np.mean(comparison.y)))


def get_categories(x,y):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['delta'] = y - x
    df['cat'] = ['control' if np.abs(df.iloc[i].delta) <= .4
                    else 'challenge' if ((df.iloc[i].delta >= 1.5) & (df.iloc[i].y < 5))
                    else 'none' for i in range(len(df))]

    chal = df[df.cat == 'challenge']
    cont = df[df.cat == 'control']
    non = df[df.cat == 'none']

    print('n challenge: ' + str(len(chal)))
    print('n control: ' + str(len(cont)))
    print('n none: ' + str(len(non)))

    f, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(cont.x,cont.y,c='b',marker='2')
    ax.scatter(chal.x,chal.y,c='r',marker='2')
    ax.scatter(non.x,non.y,c='k',alpha = .2,marker='2')
    plt.xlim(0,5)
    plt.ylim(0,)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, c='k')
    ax.spines['left'].set_bounds(15, 21)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    plt.title("Behavioral comparison \n Primates vs DCNN")

    return(df)


def get_stimlist(metrics, discard_objects=[]):
    subm = metrics[(metrics.cat != 'none') & (metrics.d_cnn > .2)]

#    discard_objects = ['9','7','4','2','0']
    discard_mask = [x not in discard_objects for x in subm.obj.values]
    subm = subm[discard_mask]
    objs, objs_counts = np.unique(subm.obj, return_counts=True)

    from scipy.spatial import distance
    from scipy.optimize import linear_sum_assignment
    from scipy import stats

    stimlist = pd.DataFrame(columns=subm.columns)
    deltas = []
    os = []
    for o in objs:
        och = subm[(subm.obj == o) & (subm.cat == 'challenge')]
        oco = subm[(subm.obj == o) & (subm.cat == 'control')]

        if len(och) < len(oco):
            nsamps = len(och)
            cA = och.copy()
            cB = oco.copy()
        else:
            nsamps = len(oco)
            cA = oco.copy()
            cB = och.copy()

        cAv = np.array(cA.d_beh.values)
        cBv = np.array(cB.d_beh.values)
        costs = np.abs(cBv[None, :] - cAv[:, None])
        row_ind, col_ind = linear_sum_assignment(costs)
        delt = np.abs(cAv[row_ind] - cBv[col_ind])

        sA = cA.iloc[row_ind]
        sB = cB.iloc[col_ind]

        #diff_d = np.abs(sA.d_km - sB.d_km)
        #min_ixs = diff_d.argsort()[:19]
        min_ixs = delt.argsort()[:len(sA)]
        sA = sA.iloc[min_ixs]
        sB = sB.iloc[min_ixs]
        delt = delt[min_ixs]

        deltas = np.append(deltas, delt)
        os = np.append(os, [o] * len(delt))
        stimlist = stimlist.append(sA, ignore_index=True)
        stimlist = stimlist.append(sB, ignore_index=True)

    return(stimlist)


def main():
    beh_metrics = pd.read_csv(cfg.beh.metrics_file)
#    plots_beh_metrics(beh_metrics)

    cnn_metrics = pd.DataFrame(load_data(cfg.cnn.metrics_svm_file))
    cnn_metrics['obj'], cnn_metrics['img'] = zip(*[tuple(map(int, x.split('_'))) for x in cnn_metrics.id])

    cnn_metrics = cnn_metrics.sort_values(by=['obj','img'])
    cnn_metrics.reset_index(inplace=True)

#    plots_cnn_metrics(cnn_metrics)

    kar_cnn = get_h5(cfg.kar.kdnn_file)
    kar_mon = get_h5(cfg.kar.kmon_file)
    kar_hum = get_h5(cfg.kar.khum_file)

    compare(kar_cnn.i1, cnn_metrics.i1, ["Kar's Alexnet", "Our Alexnet"])
    compare(kar_hum.i1, beh_metrics.i1, ["Kar's humans", "Our humans"])
    compare(kar_mon.i1, beh_metrics.i1, ["Kar's monkeys", "Our humans"])
    compare(kar_mon.i1, kar_hum.i1, ["Kar's monkeys", "Kar's humans"])

    cat_kcnn_kmon = get_categories(kar_cnn.i1,kar_mon.i1)
    cat_kcnn_khum = get_categories(kar_cnn.i1,kar_hum.i1)
    cat_kcnn_beh = get_categories(kar_cnn.i1,beh_metrics.i1)
    cat_cnn_beh = get_categories(cnn_metrics.i1, beh_metrics.i1)

    metrics = pd.DataFrame({
                            'id': cnn_metrics.id,
                            'obj': cnn_metrics.obj,
                            'img': cnn_metrics.img,
                            'd_km': kar_mon.i1,
                            'd_kh': kar_hum.i1,
                            'd_kn': kar_cnn.i1,
                            'd_cnn': cnn_metrics.i1,
                            'd_beh': beh_metrics.i1,
                            'del_km': cat_kcnn_kmon.delta,
                            'del_kh': cat_kcnn_khum.delta,
                            'del_kb': cat_kcnn_beh.delta,
                            'del': cat_cnn_beh.delta,
                            'cat_km': cat_kcnn_kmon.cat,
                            'cat_kh': cat_kcnn_khum.cat,
                            'cat_kb': cat_kcnn_beh.cat,
                            'cat': cat_cnn_beh.cat
                            })

    stimlist = get_stimlist(metrics)


    sns.set(style="whitegrid")
    ax = sns.countplot(x='obj', hue='cat', hue_order=['control','challenge'], data=stimlist)
    plt.ylim(0, 20)
    ax.set_yticks(np.arange(0, 40, 5))
    ax.set_xlabel('Objects')
    ax.set_ylabel('N')
    ax.set_title('Number of images per condition')
    ax.grid(which ='major',axis='y',alpha=.5)
    sns.despine(left=True)
    ax.legend(loc='upper left')
    plt.tight_layout()

    plt.figure()
    ax = sns.histplot(stimlist,x='d_beh',hue='cat',hue_order=['control','challenge'])
    ax.set_xlabel('Human Performance')
    ax.set_ylabel('N')

    #stimlist = stimlist[stimlist.obj != '8']
    d_chal = stimlist[stimlist.cat == 'challenge'].d_beh
    d_cont = stimlist[stimlist.cat == 'control'].d_beh
    stats.ttest_ind(d_chal,d_cont)
    stats.mannwhitneyu(d_chal,d_cont)

    reps = 34
    stim = pd.DataFrame({
            'index': np.arange(len(stimlist)),
            'id': stimlist.id,
            'obj': stimlist.obj,
            'img': stimlist.img,
            'd_km': stimlist.d_km,
            'd_kn': stimlist.d_kn,
            'd_kh': stimlist.d_kh,
            'd_cnn': stimlist.d_cnn,
            'd_beh': stimlist.d_beh,
            'cat_km': stimlist.cat_km,
            'cat_kh': stimlist.cat_kh,
            'cat_kb': stimlist.cat_kb,
            'cat': stimlist.cat
        })
    lstim = pd.concat([stim] * reps, ignore_index=True)
    out_dir = cfg.beh.out_dir
    outfile_name = 'stimlist_inhouse-metrics.csv'
    lstim.to_csv(out_dir + outfile_name)
