from project_utils import *
import config as cfg

#import glob
import pandas as pd
import numpy as np
#from scipy.stats import norm
#import itertools

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












def main():
    metrics = pd.read_csv(cfg.beh.metrics_file)

    metrics_names = ['i1','fai','hri','rt']
    suj_columns = [not x in metrics_names for x in metrics.columns.values]

    metrics_s = metrics[metrics.columns[suj_columns]]

    metrics_sl = pd.wide_to_long(metrics_s,
                                stubnames=['i1','fa','hr','rt'],
                                i="im_lab",
                                j="suj",
                                sep="_",
                                suffix='\\w+')
    metrics_sl.reset_index(inplace=True)

    plot_mean_obs(x=metrics_sl.obj, y=metrics_sl.i1, hue=metrics_sl.suj)
    plot_mean_obs(x=metrics_sl.obj, y=metrics_sl.hr, hue=metrics_sl.suj)
    plot_mean_obs(x=metrics_sl.obj, y=metrics_sl.fa, hue=metrics_sl.suj)
    plot_mean_obs(x=metrics_sl.obj, y=metrics_sl.rt, hue=metrics_sl.suj)

    plt.plot(metrics.i1)

if __name__ == '__main__':
    main()
