
import numpy as np
import matplotlib.pyplot as plt


def plot_graphs(histories, labels, n, m, mu_str, colors, linewidths, linestyles, 
                n_iters=None, threshold=1e-8, filename=None, f_star=None,
                suptitle=None, max_iter=1000):

    if f_star is None:
        f_best = min(histories[0]['func'])
        for i in range(1, len(histories)):
            f_best = min(f_best, min(histories[i]['func']))
    else:
        f_best = f_star

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    
    for i in range(len(histories)):
        resid = np.array(histories[i]['func']) - f_best
        n_iter = n_iters[i] if n_iters is not None else resid.shape[0]
        n_iter = min(n_iter, np.searchsorted(-resid, -threshold)+1)

        ax1.semilogy(resid[0:min(n_iter, max_iter)], 
                     label=labels[i], color=colors[i], 
                     linestyle=linestyles[i], linewidth=linewidths[i])
        ax2.semilogy(histories[i]['time'][0:n_iter], resid[0:n_iter], 
                     label=labels[i], linestyle=linestyles[i], color=colors[i], 
                     linewidth=linewidths[i])
    
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Func. residual', fontsize=14)
    ax2.set_xlabel('Time', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid()
    ax2.grid()
    #plt.tight_layout()
    if filename:
        print('output: %s' % filename)
        plt.savefig(filename)

        