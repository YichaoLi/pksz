import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import h5py 
import os

import utils

from scipy.interpolate import interp1d

from analysis_mcmc.my_src import get_dist
from analysis_mcmc.my_src import plot_triangle
from analysis_mcmc.my_src import plot_2d
from analysis_mcmc.my_src import plot_1d

import cosmology as cosmo
import read_data
import fitting
import models

def check_TAP_bar(result_path, result_name_list, label_list=[],
        output_path=None, output_name='', text=''):

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    color_list = ['r', 'g', 'b', 'c', 'm']

    for i in range(len(result_name_list)):
        result_name = result_name_list[i]
        c = color_list[i]
        result = h5py.File(result_path + result_name, 'r')

        t_ap_bar = result['t_ap_bar_000'][:]
        dc = result['ga_dc'][:]
        mask = result['ga_mask'][:][0]
        dc = dc[np.logical_not(mask)]

        idx = np.argsort(dc)
        dc = dc[idx]
        t_ap_bar = np.ma.array(t_ap_bar[idx])

        #if i ==1:
        #    t_ap_bar /= 2.7

        result.close()


        p = t_ap_bar >= 0
        n = t_ap_bar <= 0
        t_ap_bar.mask = n
        ax.plot(dc,  t_ap_bar, c+'-', linewidth=2.5, label=label_list[i])
        t_ap_bar.mask = p
        ax.plot(dc, -t_ap_bar, c+'--', linewidth=2.5)

        #p = t_ap_bar_y > 0
        #n = t_ap_bar_y < 0
        #ax.plot(dc_x[p],  t_ap_bar_y[p]*1.e3, 'k-')
        #ax.plot(dc_x[n], -t_ap_bar_y[n]*1.e3, 'k:')

    ax.legend(frameon=False, loc=0, title=text)
    ax.set_ylabel(r'$|\bar{T}_{\rm AP}| \,\, [\mu{\rm K}]$')
    ax.set_xlabel(r'${\rm Comoving Distance} [{\rm Mpc}/h]$')

    ax.semilogy()
    ax.set_ylim(1.e-4, 1.e2)
    ax.minorticks_on()
    ax.tick_params(length=5, width=1.5)
    ax.tick_params(which='minor', length=3, width=1.5)

    if output_path is not None:
        plt.savefig(output_path + '%s_TAP_bar.eps'%output_name, formate='eps')

def plot_jk(result_path, result_name_list, label_list=[],
        output_path=None, text='', ymin=None, ymax=None):

    for i in range(len(result_name_list)):

        fig = plt.figure(figsize=(8,4))
        ax  = fig.add_axes([0.12, 0.15, 0.83, 0.8])
    
        result_name = result_name_list[i]
        output_name = result_name.split('.')[0]
        try:
            result = h5py.File(result_path + result_name, 'r')
        except IOError:
            #print "-- %s\n-- Does not exist, pass\n"%result_name
            continue


        pksz = result['pkSZ'][:]
        if 'pkSZ_jk' in result.keys():
            pksz_random = result['pkSZ_jk'][:]
        else:
            #print "no jk samples"
            continue
        #pksz_random = result['pkSZ_random'][:]
        pksz_bins = result['pkSZ_bins'][:]

        result.close()

        pksz_cov = np.cov(pksz_random, rowvar=False, bias=True)
        spl_n = float(pksz_random.shape[0] - 1)
        pksz_cov  *= spl_n
        pksz_error = np.sqrt(pksz_cov.diagonal())

        d_bins = pksz_bins[1:] - pksz_bins[:-1]
        pksz_bins = pksz_bins[:-1] + 0.5 * d_bins

        ax.hlines(y=0, xmin=pksz_bins.min()-d_bins.min(), 
                xmax=pksz_bins.max()+d_bins.max(), 
                linestyles='--', colors='k', linewidths=2)

        ax.errorbar(pksz_bins, pksz, pksz_error,
                #np.std(pksz_random, axis=0), 
                fmt='ro', marker='o', mfc='w', mec='r', ms=7, mew=2,
                ecolor='r', elinewidth=2, capsize=0, capthick=0,
                label=label_list[i])
        for j in range(pksz_random.shape[0]):
            ax.plot(pksz_bins, pksz_random[j,:] , c='grey', 
                    linestyle='-', linewidth=0.5, )

        ax.legend(frameon=False, loc=4, title=text, ncol=2)
        ax.set_ylabel(r'$p_{\rm kSZ}\,[\mu{\rm K}]$')
        ax.set_xlabel(r'${\rm Distance}\,[{\rm Mpc}/h]$')

        ax.set_xlim(0, pksz_bins.max()+d_bins.max())
        ax.set_ylim(ymin, ymax)
        ax.minorticks_on()
        ax.tick_params(length=5, width=1.5)
        ax.tick_params(which='minor', length=3, width=1.5)

        if output_path is not None:
            plt.savefig(output_path + output_name + '_jk.eps', formate='eps')

def plot_cov(result_path, result_name_list, label_list=[],
        output_path=None, text='', jk_sample=False, cov=False):

    for i in range(len(result_name_list)):

        result_name = result_name_list[i]
        output_name = result_name.split('.')[0]
        try:
            result = h5py.File(result_path + result_name, 'r')
        except IOError:
            #print "-- %s\n-- Does not exist, pass\n"%result_name
            continue


        pksz = np.mat(result['pkSZ'][:])
        if 'pkSZ_random' in result.keys():
            pksz_random = result['pkSZ_random'][:]
            jk_sample = False
        elif 'pkSZ_jk' in result.keys():
            pksz_random = result['pkSZ_jk'][:]
            jk_sample = True
        else:
            print "Need random samples"
            exit()
        #pksz_random = result['pkSZ_random'][:]
        pksz_bins = result['pkSZ_bins'][:]

        result.close()

        if cov:
            pksz_cov = np.cov(pksz_random, rowvar=False, bias=jk_sample)
            if jk_sample:
                spl_n = float(pksz_random.shape[0] - 1)
                bin_n = pksz_cov.shape[0]
                pksz_cov  *= spl_n
        else:
            pksz_cov = np.corrcoef(pksz_random, rowvar=False)

        fig = plt.figure(figsize=(7,6))
        ax = fig.add_axes([0.10, 0.14, 0.75, 0.75])
        cax= fig.add_axes([0.83, 0.21, 0.03, 0.61])

        if cov:
            im = ax.pcolormesh(pksz_bins, pksz_bins, pksz_cov * 100, 
                    vmax=0.3, vmin=0)
        else:
            im = ax.pcolormesh(pksz_bins, pksz_bins, pksz_cov, vmax=1, vmin=0)
        fig.colorbar(im, ax=ax, cax=cax)

        ax.set_xlim(xmin=pksz_bins.min(), xmax=pksz_bins.max())
        ax.set_ylim(ymin=pksz_bins.min(), ymax=pksz_bins.max())
        ax.set_xlabel(r'${\rm Distance}\, [{\rm Mpc}/h]$')
        ax.set_ylabel(r'${\rm Distance}\, [{\rm Mpc}/h]$')
        ax.set_aspect('equal')
        ax.minorticks_on()
        ax.tick_params(length=5, width=1.5, direction='out')
        ax.tick_params(which='minor', length=3, width=1.5, direction='out')
        ax.set_title(text.replace('\n', ',') + ', ' + label_list[i])

        if cov:
            cax.set_ylabel(r'${\rm Covariance\, Matrix}\,[\mu{\rm K}^2]\times10^{-2}$')
        else:
            cax.set_ylabel(r'${\rm Correlation\, Matrix}$')

        cax.minorticks_on()
        cax.tick_params(length=3, width=1.5, direction='out')
        cax.tick_params(which='minor', length=2, width=1.5, direction='out')

        if cov:
            plt.savefig(output_path + output_name + '_covm.pdf', formate='pdf')
        else:
            plt.savefig(output_path + output_name + '_corm.pdf', formate='pdf')

def plot_fitting_result_vs_ap(fitting_result, ap_list, label_list, 
        output_path=None, output_name=''):
    '''
    fitting_result in shape of len(ap_list) x len(label_list) x len(result)
    result = [tau, sig_plus, sig_mins, snr_plus, snr_mins]
    '''


    fig = plt.figure(figsize=(6,5))
    ax  = fig.add_axes([0.15, 0.15, 0.7, 0.8])
    ax2 = ax.twinx()
    color_list = ['r', 'g', 'b', 'c', 'm']


    n_ap, n_measure, n_result = fitting_result.shape

    ap_shift = np.arange(n_measure) - 0.5*(n_measure-1)
    ap_list = np.array(ap_list)

    for i in range(n_measure):

        c = color_list[i]
        
        data = fitting_result[:,i,:]
        np.ma.masked_invalid(data)
        x = ap_list + ap_shift[i] * (0.3 / float(n_measure))

        ax2.plot(x, data[:,6], c+'.--', linewidth=1.5, drawstyle='steps-mid')

        ax.errorbar(x, data[:,0], data[:,1:3].T, 
                fmt=c+'o', marker='s', mfc='w', mec=c, ms=6, mew=2,
                ecolor=c, elinewidth=2, capsize=3, capthick=2,
                label=label_list[i])

    ax.set_xlabel('AP Size [arcmin]')
    ax.set_ylabel(r'$\bar{\tau} \times 10^{-4}$')

    ax2.set_ylabel(r'$\chi^2/{\rm d.o.f.}$', rotation = 270,  labelpad=25)
    #ax2.set_ylabel(r'$\bar{\tau}/\sigma(\bar{\tau})$')
    #ax2.set_ylabel(r'$\chi^2_{\rm NULL}$')
    ax2.set_ylim(ymax=4.)

    ax.legend()
    plt.savefig(output_path + output_name + '_chisq_ap.eps', formate='eps')

    cols = 'c' + '|cc' * n_measure

    tex_file = open(output_path + output_name + '_chisq_ap.tex', 'w')
    tex_file.write(r'\begin{tabular}{%s} \hline\hline'%cols + '\n')
    for i in range(n_measure-1):
        tex_file.write(r'   &\multicolumn{2}{c|}{%s}'%label_list[i] + '\n')
    tex_file.write(r'   &\multicolumn{2}{c}{%s} \\'%label_list[-1] + '\n\n')

    tex_file.write(r'   $\theta_{\rm AP}[{\rm arcmin}]$' + '\n')
    for i in range(n_measure-1):
        tex_file.write(r'   & $\bar{\tau}[\times10^{-4}]$ & S/N' + '\n')
    tex_file.write(r'   & $\bar{\tau}[\times10^{-4}]$ & S/N \\\hline' + '\n\n')
    for j in range(n_ap):
        tex_file.write(r'   $%d$'%ap_list[j] + '\n')
        for i in range(n_measure-1):
            tex_file.write(r'   & $%3.2f\pm%3.2f$ & $%3.2f\sigma$'\
                    %(fitting_result[j, i, 0], 
                      fitting_result[j, i, 1],
                      fitting_result[j, i, 3]) + '\n')
        tex_file.write(r'   & $%3.2f\pm%3.2f$ & $%3.2f\sigma$ \\'\
                    %(fitting_result[j, -1, 0], 
                      fitting_result[j, -1, 1],
                      fitting_result[j, -1, 3]) + '\n\n')


    tex_file.write(r'   \hline\hline' + '\n')
    tex_file.write(r'\end{tabular}' + '\n')
    tex_file.close()


def plot_chisq_vs_ap(result_path, result_name_list, ap_list, label_list=[],
        output_path=None, output_name=''):

    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_axes([0.12, 0.15, 0.8, 0.8])
    color_list = ['r', 'g', 'b', 'c', 'm']

    for j in range(len(result_name_list)):
        name = result_name_list[j]
        c = color_list[j]
        label = label_list[j]
        chisq_list = []
        for i in range(len(ap_list)):

            #print result_path + name + '_AP%darcm.h5'%ap_list[i]
            #if not os.path.exists(result_path + name + '_AP%darcm.h5'%ap_list[i]):
            #    chisq_list.append(-999)
            #    continue
            result_file = result_path + name + '_AP%darcm.h5'%ap_list[i]
            print result_file
            try:
                result = h5py.File(result_file, 'r')

                pksz = np.mat(result['pkSZ'][:])
                pksz_random = result['pkSZ_random'][:]
                pksz_bins = result['pkSZ_bins'][:]

                result.close()
            except IOError:
                print "File Does Not Exist"
                chisq_list.append(-999)
                continue


            pksz_mean = np.mean(pksz_random, axis=0)
            pksz_covm = pksz_random - pksz_mean[None, :]
            pksz_covm = pksz_covm[:, :, None] * pksz_covm[:, None, :]
            pksz_covm = np.mean(pksz_covm, axis=0)

            chisq = pksz * np.mat(pksz_covm).I * pksz.T / float(pksz.shape[1])
            chisq = chisq[0,0] ** 0.5
            chisq_list.append(chisq)

        chisq_list = np.ma.array(chisq_list)
        chisq_list[chisq_list==-999] = np.ma.masked
        #print chisq_list
        ax.plot(ap_list, chisq_list, c+'o-', linewidth=2, label=label)
        #ax.scatter(ap_list, chisq_list, s=20, c=c, linewidths=2, label=label,
        #        marker='o')

    #ax.set_xlim(xmin=0, xmax=max(ap_list))
    ax.set_xlabel('AP Size [arcmin]')
    ax.set_ylabel(r'$\chi^2_{\rm NULL}$')
    ax.legend()
    plt.savefig(output_path + output_name + '_chisq_ap.eps', formate='eps')
    #plt.savefig(output_path + output_name + '_chisq_ap.png', formate='png')
    #plt.savefig(output_path + output_name + '_chisq_ap.pdf', formate='pdf')

def plot_error(result_path, result_name_list, label_list=[], 
        output_path=None, output_name='', bins_shift=0.5, text='', 
        ymax=None, ymin=None):

    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_axes([0.12, 0.15, 0.78, 0.8])
    color_list = ['r', 'g', 'b', 'c', 'm']
    
    ax2 = ax.twinx()
    
    #fig.text(0.7, 0.18, text)

    result_numb = len(result_name_list)
    bins_shift = np.arange(result_numb) - 0.5*(result_numb-1)

    for i in range(len(result_name_list)):
        result_name = result_name_list[i]
        c = color_list[i]
        result = h5py.File(result_path + result_name, 'r')

        pksz = result['pkSZ'][:]
        pksz_random = result['pkSZ_random'][:]
        pksz_numb = result['pkSZ_numb'][:]
        pksz_bins = result['pkSZ_bins'][:]


        d_bins = pksz_bins[1] - pksz_bins[0]
        pksz_bins = pksz_bins[:-1] + 0.5 * d_bins
        print pksz_bins
        pksz_bins += bins_shift[i] * (0.5 * d_bins / float(result_numb))

        result.close()

        #if i == 0:
        #    ax.hlines(y=0, xmin=pksz_bins.min()-d_bins, xmax=pksz_bins.max()+d_bins, 
        #            linestyles='--', colors='k', linewidths=2)

        #print np.mean(pksz / np.std(pksz_random, axis=0))

        #ax.step(pksz_bins, np.std(pksz_random, axis=0), where='mid', linewidth=4,
        #        color=c, label=label_list[i])
        ax.plot(pksz_bins, np.std(pksz_random, axis=0), c+'o:', 
                mec=c, ms=5,
                linewidth=2, drawstyle='steps-mid', label=label_list[i])
        ax2.step(pksz_bins, pksz_numb, where='mid', linewidth=2,
                color=c)
        #ax.errorbar(pksz_bins, pksz, np.std(pksz_random, axis=0), 
        #        fmt=c+'o', marker='o', mfc='w', mec=c, ms=7, mew=2,
        #        ecolor=c, elinewidth=2, capsize=0, capthick=0,
        #        label=label_list[i])


    ax.legend(frameon=False, loc=1, title=text)
    ax.set_ylabel(r'$\sigma_{p_{\rm kSZ}}\,[\mu{\rm K}]$')
    ax.set_xlabel(r'${\rm Distance}\,[{\rm Mpc}/h]$')

    ax.set_xlim(0, pksz_bins.max()+d_bins)
    ax.set_ylim(ymin, ymax)
    ax.minorticks_on()
    ax.tick_params(length=5, width=1.5)
    ax.tick_params(which='minor', length=3, width=1.5)

    ax2.set_ylabel(r'Number of Galaxy Pair')

    #ax2.set_xlim(0, pksz_bins.max()+d_bins)
    ax2.semilogy()
    ax2.minorticks_on()
    ax2.tick_params(length=5, width=1.5)
    ax2.tick_params(which='minor', length=3, width=1.5)
    if output_path is not None:
        plt.savefig(output_path + '%s_pkszErr.eps'%output_name, formate='eps')


def plot_pkSZ_ap(result_path, result_name_list, ap_list, label_list=[], 
        output_path=None, output_name='', ymax=None, ymin=None, xmax=150, camb=None, 
        logmass_list=None):

    ap_n = len(ap_list)
    fg_w = 6.
    fg_h = 1.6 * ap_n
    ax_h = 1.5
    sp_h = 0.0
    b = 0.8 * (1. - (ap_n * ax_h + (ap_n-1) * sp_h)/fg_h)
    fig = plt.figure(figsize=(fg_w, fg_h))
    l = 0.13
    w = 0.82
    h = ax_h / fg_h

    fitting_result = []
    for i in range(ap_n):

        bi = b + (ap_n - i - 1) * (ax_h + sp_h) / fg_h
        ax = fig.add_axes([l, bi, w, h])

        result_name_list_ap = [x%ap_list[i] for x in result_name_list]
        result = plot_pkSZ(result_path, result_name_list_ap, label_list=label_list,
            output_path=None, output_name='',
            camb=camb, logmass_list = logmass_list,
            ymax=ymax, ymin=ymin, ax=ax) 
        fitting_result.append(result)

        text = r'$\theta_{\rm AP} = %d\,{\rm arcmin}$'%ap_list[i]
        ax.text(110, 0.06, text, fontsize='medium')

        if i == 0:
            ax.legend(frameon=False, loc=4, ncol=1, fontsize='small')
        ax.set_ylabel(r'$p_{\rm kSZ}\,[\mu{\rm K}]$')
        if i == ap_n - 1:
            ax.set_xlabel(r'${\rm Distance}\,[{\rm Mpc}/h]$')
        else:
            ax.set_xticklabels([])

        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, ymax)

        ax.minorticks_on()
        ax.tick_params(length=5, width=1.5, direction='in')
        ax.tick_params(which='minor', length=3, width=1.5, direction='in')

        plot_cov(result_path, result_name_list_ap, label_list=label_list,
            output_path=output_path, text=text)
        plot_cov(result_path, result_name_list_ap, label_list=label_list,
            output_path=output_path, text=text, cov=True)

    if output_path is not None:
        plt.savefig(output_path + '%s_combined.eps'%output_name, formate='eps')

    fitting_result = np.array(fitting_result)
    plot_fitting_result_vs_ap(fitting_result, ap_list, label_list,
            output_path=output_path, output_name=output_name)

def plot_pkSZ(result_path, result_name_list, label_list=[],
        output_path=None, output_name='', bins_shift=0.5, text='', 
        ymax=None, ymin=None, camb=None, logmass_list=None, jk_sample=False, 
        ax = None):

    sub_plot = True
    if ax is None:
        fig = plt.figure(figsize=(8,4))
        ax  = fig.add_axes([0.12, 0.15, 0.83, 0.8])
        sub_plot = False
    color_list = ['r', 'g', 'b', 'c', 'm']
    
    #fig.text(0.7, 0.18, text)

    result_numb = len(result_name_list)
    bins_shift = np.arange(result_numb) - 0.5*(result_numb-1)

    fitting_result = []

    hline_plot = True

    for i in range(len(result_name_list)):
        result_name = result_name_list[i]
        c = color_list[i]
        try:
            result = h5py.File(result_path + result_name, 'r')
        except IOError:
            print "x  %s\n   Does not exist, pass\n"%result_name
            fitting_result.append([np.inf, ]*7)
            continue

        pksz = result['pkSZ'][:]
        if 'pkSZ_random' in result.keys():
            pksz_random = result['pkSZ_random'][:]
            jk_sample = False
        elif 'pkSZ_jk' in result.keys():
            pksz_random = result['pkSZ_jk'][:]
            jk_sample = True
        else:
            print "Need random samples"
            exit()

        #if jk_sample:
        #    pksz_random = result['pkSZ_jk'][:]
        #else:
        #    pksz_random = result['pkSZ_random'][:]
        pksz_bins = result['pkSZ_bins'][:]


        #print pksz_bins
        d_bins = pksz_bins[1:] - pksz_bins[:-1]
        pksz_bins_raw = pksz_bins[:-1] + 0.5 * d_bins
        #print pksz_bins
        pksz_bins = pksz_bins_raw + bins_shift[i] * (0.5 * d_bins / float(result_numb))

        result.close()

        lin_scale = pksz_bins_raw > 25.

        #if vij_result is not None:

        #    results = h5py.File(vij_result[i], 'r')
        #    vij = results['vij'][0, 0, :]
        #    rh  = results['rh'][:]
        #    vij_func = interp1d(rh, vij, kind='slinear')
        if camb is not None:
            vij, rh = models.mean_pairwise_v(camb, rh=pksz_bins_raw[lin_scale],
                    logmass_min=logmass_list[i][0], logmass_max=logmass_list[i][1], )
            vij = vij.flatten()
            vij_p, rh_p = models.mean_pairwise_v(camb, rh=np.linspace(10, 150, 100),
                    logmass_min=logmass_list[i][0], logmass_max=logmass_list[i][1], )
            vij_p = vij_p.flatten()

            #camb.params['w'] = -2.
            #vij2, rh2 = models.mean_pairwise_v(camb, rh=pksz_bins_raw[lin_scale],
            #        logmass_min=logmass_list[i][0], logmass_max=logmass_list[i][1], )
            #vij2 = vij2.flatten()
            #camb.params['w'] = -1.

        pksz_cov = np.cov(pksz_random, rowvar=False, bias=jk_sample)
        if jk_sample:
            spl_n = float(pksz_random.shape[0] - 1)
            bin_n = pksz_cov.shape[0]
            pksz_cov  *= spl_n
            pksz_covi  = np.linalg.inv(pksz_cov[:, lin_scale][lin_scale, :])
            pksz_covi *= (spl_n - bin_n) / spl_n
        else:
            pksz_covi = np.linalg.inv(pksz_cov[:, lin_scale][lin_scale, :])
        pksz_error = np.sqrt(pksz_cov.diagonal())
        amp, amp_upper, amp_lower, factor, chisq_null, chisq = \
                fitting.amp_fitting( pksz[lin_scale], vij, pksz_covi = pksz_covi)
                #pksz_err=np.std(pksz_random, axis=0)[lin_scale])
        val_best = amp/factor*1.e4
        sig_plus = amp_upper/factor*1.e4 - amp/factor*1.e4
        sig_mins = amp/factor*1.e4 - amp_lower/factor*1.e4
        snr_plus = val_best/sig_plus
        snr_mins = val_best/sig_mins
        fitting_result.append(
                [val_best, sig_plus, sig_mins, snr_plus, snr_mins, chisq_null, chisq])
        print "o  %s"%result_name
        print "   tau = %4.2f +%4.2f/-%4.2f (+%4.2f/-%4.2f)sigma"%\
                (val_best, sig_plus, sig_mins, snr_plus, snr_mins)
        print "   chisq_null = %4.2f\n"%chisq_null
        #(val_best, sig_plus, sig_mins, val_best/sig_plus, val_best/sig_mins)

        #amp2, amp_upper2, amp_lower2, factor2  = \
        #        fitting.amp_fitting( pksz[lin_scale], vij2, 
        #        pksz_cov = np.cov(pksz_random[:, lin_scale], rowvar=False))
        #        #pksz_err=np.std(pksz_random, axis=0)[lin_scale])
        #val_best2 = amp2/factor2*1.e4
        #sig_plus2 = amp_upper2/factor2*1.e4 - amp2/factor2*1.e4
        #sig_mins2 = amp2/factor2*1.e4 - amp_lower2/factor2*1.e4
        #print "tau = %4.2f +%4.2f/-%4.2f (+%4.2f/-%4.2f)sigma"%(
        #        val_best2, sig_plus2, sig_mins2, val_best2/sig_plus2, val_best2/sig_mins2)

        if hline_plot:
            ax.hlines(y=0, xmin=pksz_bins.min()-d_bins.min(), 
                    xmax=pksz_bins.max()+d_bins.max(), 
                    linestyles='--', colors='k', linewidths=2)
            hline_plot = False

        #print np.mean(pksz / np.std(pksz_random, axis=0))

        #ax.plot(pksz_bins, np.mean(pksz_random, axis=0), c+'.--', linewidth=2)
        ax.errorbar(pksz_bins, pksz, pksz_error,
                #np.std(pksz_random, axis=0), 
                fmt=c+'o', marker='o', mfc='w', mec=c, ms=5, mew=1.5,
                ecolor=c, elinewidth=2, capsize=0, capthick=0,
                label=label_list[i])
        if not sub_plot:
            ax.plot(rh_p, vij_p * amp, c+'-', linewidth=1.5, 
                label=r'$\bar{\tau}=%3.2f^{+%3.2f}_{-%3.2f}\times10^{-4}$'%( 
                    amp/factor*1.e4, (amp_upper-amp)/factor*1.e4, 
                    (amp-amp_lower)/factor*1.e4))
        else:
            ax.plot(rh_p, vij_p * amp, c+'-', linewidth=1.5)

        #ax.plot(rh2, vij2 * amp2, c+'--', linewidth=2)

    print '='*20

    if not sub_plot:
        ax.legend(frameon=False, loc=4, title=text, ncol=2)
        ax.set_ylabel(r'$p_{\rm kSZ}\,[\mu{\rm K}]$')
        ax.set_xlabel(r'${\rm Distance}\,[{\rm Mpc}/h]$')

        ax.set_xlim(0, pksz_bins.max()+d_bins.max())
        ax.set_ylim(ymin, ymax)
        ax.minorticks_on()
        ax.tick_params(length=5, width=1.5)
        ax.tick_params(which='minor', length=3, width=1.5)

    if output_path is not None:
        plt.savefig(output_path + '%s_pksz.eps'%output_name, formate='eps')
    return fitting_result

def ga_numb_hist(ga):

    fig = plt.figure(figsize=(8,4))
    ax  = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    hist, bins = np.histogram(ga.catalog['DC'], bins=20)
    hist = np.append(hist, [0,])
    ax.step(bins, hist, where='post')

    hist, bins = np.histogram(ga.catalog['DC'][:70000], bins=bins)
    hist = np.append(hist, [0,])
    ax.step(bins, hist, where='post')

    hist, bins = np.histogram(ga.catalog['DC'][70000:140000], bins=bins)
    hist = np.append(hist, [0,])
    ax.step(bins, hist, where='post')

def check_map():

    import copy

    data_path = '/data/ycli/6df/'
    ga = read_data.GAMA_CAT(data_path)
    ga.catalog   = '6df_ra_dec_z_6dFGSzDR3.cat'

    #data_path = '/data/ycli/cgc/'
    #ga = read_data.GAMA_CAT(data_path)
    #ga.catalog   = 'CGC_ra_dec_redshift.dat'

    #data_path = '/data/ycli/gama/'
    #ga = read_data.GAMA_CAT(data_path)
    #ga.catalog   = 'GAMA_DistancesFrames_ApMatchedCat.dat'

    c = cosmo.Cosmology()
    c = c.init_physical(ombh2=0.02230, omch2=0.1188, H0=67.74, omkh2=0.00037)
    ga.est_comoving_dist(c)
    ga.get_ga_coord()

    #ga_numb_hist(ga)

    data_path = '/data/ycli/cmb/'
    pl = read_data.PLANCK_MAP(data_path)
    pl.mask      = 'masks.fits'
    pl.kSZ_map   = 'DX11d_2DILC_MAP_HFIonly_NOPS.fits'
    pl.kSZ_noise = 'DX11d_2DILC_NOISE_HFIonly_NOPS.fits'

    l, b = utils.convert_eq2ga(ga.catalog['RA'], ga.catalog['DEC'])


    pl.check_map(ga.catalog, ra=ga.catalog['L'], dec=ga.catalog['B'])


def check_cij():

    dc_delta = np.linspace(0, 500, 100)
    dc_i = np.ones(100) * 100
    dc_j = np.ones(100) * 100 + dc_delta
    cos_theta = np.linspace(-1, 1, 100)

    dc_i = dc_i[:,None]
    dc_j = dc_j[:,None]
    cos_theta = cos_theta[None, :]

    cij = np.sqrt(dc_i*dc_i + dc_j*dc_j - 2.*dc_i*dc_j*cos_theta)
    cij[cij==0] = np.inf
    cij = (dc_j - dc_i)*(1. + cos_theta) / 2./ cij

    #plt.pcolormesh(cos_theta.flatten(), dc_delta, cij)
    plt.plot(dc_delta, np.mean(cij, axis=1))
    #plt.show()

def plot_m_hist(data_list, label_list, z_list=None, output='./m_hist.eps'):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.14, 0.13, 0.8, 0.8])
    color_list = 'krbgc'

    zmin = 15.5
    zmax = 22.5
    
    if z_list is None: 
        z_list = ['MODELFLUX_r', 'MODELFLUX_r', 'MODELFLUX_r', 'MODELFLUX_r']
    
    for i in range(len(data_list)):
        data_name = data_list[i]
        label = label_list[i]
        z = z_list[i]
        c = color_list[i]
    
        data = np.genfromtxt(data_name, names=True)
        
        data_sele = np.isfinite(data[z])
        data = data[data_sele]
        print data.dtype
        data[z] = 22.5 - 2.5 * np.log10(data[z])
        gl_n = data.shape[0]
        
        bins = np.linspace(zmin, zmax, 50)
        #if zmin is not None:
        #    if logspace:
        #        bins = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        #    else:
        #        bins = np.linspace(zmin, zmax, 100)
        #else:
        #    bins = 100
        
        hist, bins = np.histogram(data[z], bins=bins)
        
        ax.step(bins[:-1], hist/1.e4, where='pre', linewidth=2, color=c, 
                label='$%9d$ %s'%(gl_n, label))

    ax.legend(frameon=False, loc=2)
    ax.set_ylabel(r'Number $\times10^{4}$')
    ax.set_xlabel('%s'%z_list[0])

    ax.set_xlim(zmin, zmax)
    ax.set_ylim(0, 7.99)
    ax.minorticks_on()
    ax.tick_params(length=5, width=1.5)
    ax.tick_params(which='minor', length=3, width=1.5)
    
    plt.savefig(output, formate='eps')
    plt.show()


def plot_z_hist(data_list, label_list, z_list=None, output=None):

    fig = plt.figure(figsize=(5,4))
    ax  = fig.add_axes([0.14, 0.13, 0.8, 0.8])
    color_list = 'krbgcm'
    
    if z_list is None: 
        z_list = ['Z', 'Z', 'Z', 'Z', 'Z'] #, 'Z_TONRY']
    zmin = 0.0
    zmax = 0.5
    
    for i in range(len(data_list)):
        data_name = data_list[i]
        label = label_list[i]
        z = z_list[i]
        c = color_list[i]
    
        data = np.genfromtxt(data_name, names=True)
        data_n = data.shape[0]
        
        #z_sel = np.isfinite(data[z])
        z_sel = np.logical_and(data['Z'] > 0.01, data['Z'] < 0.8)
        
        if 'CGC' in data.dtype.names:
            cgc = data['CGC'] >= 0
            if 'MC_MATCH' in data.dtype.names:
                has_mass = data['MC_MATCH'] != 0
                cgc_has_mass = np.logical_and(cgc, has_mass)
            else:
                cgc_has_mass = cgc
        else:
            cgc_has_mass = np.ones(data_n).astype('bool')

        data_cgc = data[np.logical_and(cgc_has_mass, z_sel)]
        gl_n = data_cgc.shape[0]

        bins = np.arange(zmin, zmax, 0.01)
        
        hist, bins = np.histogram(data_cgc[z], bins=bins)
        
        #ax.step(bins[:-1], hist/1.e4, where='pre', linewidth=2, color=c, 
        #        label='$%9d$ %s'%(gl_n, label))
        ax.plot(bins[:-1], hist/1.e4, c+'-', linewidth=2, drawstyle='steps-post', 
                label='$[%9d]$ %s'%(gl_n, label))

    ax.legend(frameon=False, loc=0)
    #ax.set_ylabel(r'Number $\times10^{4}$')
    ax.set_ylabel(r'Sample Amount $\times10^{4}$')
    ax.set_xlabel('%s'%z_list[0])

    ax.set_xlim(zmin, zmax)
    #ax.set_ylim(0, 9.9)
    ax.minorticks_on()
    ax.tick_params(length=5, width=1.5)
    ax.tick_params(which='minor', length=3, width=1.5)
    
    if output is not None:
        plt.savefig(output, formate='eps')

def plot_bias_multi(data_path, data_file, label_list, z_indx=0):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.13, 0.13, 0.75, 0.8])
    ax2 = ax.twinx()
    c = 'rbgck'

    for i in range(len(data_file)):

        data = np.loadtxt(data_path + data_file[i]%'b1')
        kh = data[0,:][1:]
        b = data[z_indx+1, :][1:]
        z = data[z_indx+1, :][0]
        ax.plot(kh, b, c[i]+'-', linewidth=2, label=label_list[i])

        data = np.loadtxt(data_path + data_file[i]%'b2')
        kh = data[0,:][1:]
        b = data[z_indx+1, :][1:]
        ax2.plot(kh, b, c[i]+'--', linewidth=2)

    ax.legend(title='Redshift $z=%3.1f$'%z)
    ax.semilogx()
    ax.set_xlabel(r'$k\,[h/{\rm Mpc}]$')
    ax.set_ylabel(r'$b_{\rm halo}^{(1)}$')
    ax.set_ylim(0.5, 2.1)

    ax2.set_ylabel(r'$b_{\rm halo}^{(2)}$', rotation=270, labelpad=18)
    ax2.set_ylim(0.1, 5.9)


def plot_bias(b1, b2, kh, redshifts, b1_old, b2_old):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.13, 0.13, 0.7, 0.8])
    ax2 = ax.twinx()
    c = 'rbgck'

    for i in range(b1.shape[0]):

        ax.plot(kh, b1[i,:], c[i]+'-', linewidth=1, label='$z = %6.4f $'%redshifts[i])
        ax2.plot(kh, b2[i,:], c[i]+'--', linewidth=1)

        ax.plot(kh, b1_old[i,:], c[i]+':', linewidth=2, label='$z = %6.4f $'%redshifts[i])
        ax2.plot(kh, b2_old[i,:], c[i]+':', linewidth=2)

    ax.legend()
    ax.semilogx()
    ax.set_xlabel(r'$k\,[h/{\rm Mpc}]$')
    ax.set_ylabel(r'$b_{\rm halo}^{(1)}$')
    ax.set_ylim(0.5, 2.1)

    ax2.set_ylabel(r'$b_{\rm halo}^{(2)}$', rotation=270, labelpad=15)
    ax2.set_ylim(0.1, 5.9)


def plot_growth(growth_fact, growth_rate, kh, redshifts):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.13, 0.13, 0.7, 0.8])
    ax2 = ax.twinx()
    c = 'rbgck'

    for i in range(growth_fact.shape[0]):

        ax.plot(kh, growth_fact[i,:], c[i]+'-', linewidth=2, 
                label='$z = %6.4f $'%redshifts[i])
        ax2.plot(kh, growth_rate[i,:], c[i]+'--', linewidth=2)

    ax.legend()
    ax.semilogx()
    ax.set_xlabel(r'$k\,[h/{\rm Mpc}]$')

    ax.set_ylabel(r'Growth Factor $D(a(z), k)$')
    ax2.set_ylabel(r'Growth Rate $a(z)\,{\rm d}\ln D(a(z), k) / {\rm d} \ln a(z)$',
            rotation=270, labelpad=15)


def plot_growth_factor(growth_factor, growth_rate, kh, z, 
        growth_interp, growth_rate_interp, redshifts):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.14, 0.13, 0.7, 0.8])
    ax2 = ax.twinx()
    c = 'rbgck'

    for i in range(growth_factor.shape[0]):

        #ax.plot(z, growth_factor[i,:] / growth_factor[0,:], c[i]+'-', linewidth=2, 
        #        label='$k = %6.4f $'%kh[i] + r'$\,[h/{\rm Mpc}]$')
        #ax2.plot(z, growth_rate[i,:] / growth_rate[0,:], c[i]+'--', linewidth=2)

        #ax.plot(1./(1.+z), growth_factor[i,:], c[i]+'-', linewidth=2, 
        #        label='$k = %6.4f $'%kh[i] + r'$\,[h/{\rm Mpc}]$')
        #ax2.plot(1./(1.+z), growth_rate[i,:], c[i]+'--', linewidth=2)

        ax.plot(z, growth_factor[i,:], c[i]+'-', linewidth=2, 
                label='$k = %6.4f $'%kh[i] + r'$\,[h/{\rm Mpc}]$')
        ax2.plot(z, growth_rate[i,:], c[i]+'--', linewidth=2)

        ax.plot(redshifts, growth_interp[i,:], c[i]+'o')
        ax2.plot(redshifts, growth_rate_interp[i,:], c[i]+'s')

    ax.legend()
    #ax.semilogx()
    #ax.set_xlabel(r'$k\,[h/{\rm Mpc}]$')
    ax.set_xlabel(r'$z$')

    #ax.set_ylabel(r'Growth Factor $D(z, k)/D(z, ' + 'k=%6.4f'%kh[0] + ')$')
    #ax2.set_ylabel(r'Growth Rate $f(z, k)/f(z, ' + 'k=%6.4f'%kh[0] + ')$')

    ax.set_ylabel(r'Growth Factor $D(a(z), k)$')
    ax2.set_ylabel(r'Growth Rate $a(z)\,{\rm d}\ln D(a(z), k) / {\rm d} \ln a(z)$')

def plot_PK(pk, kh, z, pk_2, kh_2):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.14, 0.13, 0.8, 0.8])
    c = 'rbgck'

    for i in range(pk.shape[0]):

        ax.plot(kh, pk[i,:], c[i]+'-', linewidth=1, label='$z = %3.2f$'%z[i])
        ax.plot(10**kh_2, pk_2[i,:], c[i]+':', linewidth=2, label='$z = %3.2f$'%z[i])

    #data = np.loadtxt('./data/Tinker_hmf/kVector_PLANCK-SMT.txt')
    #ax.plot(data[:,0], data[:,1], 'k:', linewidth=2)

    #pkf = lambda lnk, i: np.interp(lnk, np.log(kh), pk[i])
    #lnk = np.linspace(np.log(1.e-4), np.log(50), 200)
    #ax.plot(np.exp(lnk), pkf(lnk, 0), 'k-', linewidth=1.5)

    #pkf = lambda k, i: np.interp(k, kh, pk[i])
    #k = np.linspace(1.e-4, 100, 5000)
    #ax.plot(k, pkf(k, 0), 'k-', linewidth=1.5)

    ax.legend()
    ax.loglog()
    ax.set_xlabel(r'$k\,[h/{\rm Mpc}]$')
    ax.set_ylabel(r'$P(k)\,[{\rm Mpc}/h]^3$')

    #plt.show()

def plot_sigma_sq(mass, sig_sq, z, sig_sq_old):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.14, 0.13, 0.8, 0.8])
    c = 'rbgck'

    for i in range(sig_sq.shape[0]):

        ax.plot(mass, sig_sq[i,:], c[i]+'-', linewidth=1, label='$z = %3.2f$'%z[i])
        ax.plot(mass, sig_sq_old[i,:], c[i]+':', linewidth=2, label='$z = %3.2f$'%z[i])

    ax.legend()
    ax.loglog()
    ax.set_xlabel(r'$M\,[M_{\odot}/h]$')
    ax.set_ylabel(r'$\sigma^2(M, z)$')


def plot_vij(vij, rh, z):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.15, 0.13, 0.8, 0.8])
    c = 'rbgck'

    for i in range(vij.shape[0]):

        ax.plot(rh, -vij[i,:], c[i]+'-', linewidth=2, label='$z = %3.2f$'%z[i])

    ax.legend()
    #ax.semilogy()
    ax.set_xlabel(r'$r\,[{\rm Mpc}/h]$')
    ax.set_ylabel(r'Pairwise Velosity $-v_{ij}(r)\, [{\rm km/s}]$')


def plot_XI(xi, rh, z, volume_averaged=False):

    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.15, 0.13, 0.8, 0.8])
    c = 'rbgck'

    for i in range(xi.shape[0]):
        xi_tmp = np.ma.array(xi[i,:])
        xi_tmp[xi_tmp<=0] = np.ma.masked
        ax.plot(rh, xi_tmp, c[i]+'-', linewidth=2, label='$z = %3.2f$'%z[i])

        xi_tmp = np.ma.array(xi[i,:])
        xi_tmp[xi_tmp>=0] = np.ma.masked
        ax.plot(rh, -xi_tmp, c[i]+'--', linewidth=2)

    ax.legend()
    ax.semilogy()
    ax.set_xlabel(r'$r\,[{\rm Mpc}/h]$')
    if volume_averaged:
        ax.set_ylabel(r'$\bar{\xi}(r)$')
    else:
        ax.set_ylabel(r'$\xi(r)$')

    #plt.show()

def plot_dndm(dndm, mass, z, rhom_mean, sigma_sq=None, output='./'):

    fig = plt.figure(figsize=(6,4))
    if sigma_sq is not None:
        ax  = fig.add_axes([0.14, 0.14, 0.70, 0.80])
        ax2 = ax.twinx()
    else:
        ax  = fig.add_axes([0.14, 0.14, 0.8, 0.80])
    c = 'rbgck'

    print dndm.shape

    for i in range(dndm.shape[0]):

        #ax.plot(np.log10(mass), np.log10(mass**2./rhom_mean[i,:] * dndm[i,:]), 
        #        c[i]+'-', linewidth=2, label='$z = %3.2f$'%z[i])
        ax.plot(np.log10(mass), np.log10(dndm[i,:]), 
                c[i]+'-', linewidth=2, label='$z = %3.2f$'%z[i])
        if sigma_sq is not None:
            ax2.plot(np.log10(mass), np.log10(1./np.sqrt(sigma_sq[i,:])), c[i]+'--', 
                    linewidth=2, label='$z = %3.2f$'%z[i])
        #    ax2.plot(1./np.sqrt(sigma_sq[i,:]), mass**2./rhom_mean[i,:] * dndm[i,:], 
        #            c[i]+'-', linewidth=2)

    #data = np.loadtxt('./data/Tinker_hmf/mVector_PLANCK-SMT.txt')
    #ax.plot(np.log10(data[:,0]), np.log10(data[:,5]), 'k--', linewidth=2)

    #if sigma_sq is not None:
    #    ax2.plot(np.log10(data[:,0]), np.log10(1./data[:,1]), 'k--', linewidth=2)

    ax.legend(title=r'${\rm d}n/{\rm d}M$')
    #ax.loglog()
    ax.set_xlabel(r'$M\,[M_{\odot}/h]$')
    ax.set_ylabel(r'$(M^2/\bar{\rho}_{\rm m})\,{\rm d}n/{\rm d}M$')

    if sigma_sq is not None:
        #ax2.loglog()
        #ax2.semilogx()
        #ax2.legend(title=r'$\log(1/\sigma)$')
        ax2.set_ylabel(r'$\log(1/\sigma)$')
        #ax2_xticklabels = np.log10(1./np.sqrt(sigma_sq))
        #ax2_xticklabels = np.round(ax2_xticklabels, decimals=2).astype('str')
        #print ax2_xticklabels
        #print ax2_xticklabels.dtype
        #ax2.set_xticks(sigma_sq_lo)
        #ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        #ax2.set_xticklabels(ax2_xticklabels)
        #ax2.set_xlabel(r'$1/\sigma$')


    plt.savefig(output + 'dndM.eps', formate='eps')

    #plt.show()

def plot_f_sigma(f_sigma, sigma_sq, z, output='./'):
    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.80])
    c = 'rbgck'
    for i in range(f_sigma.shape[0]):
        ax.plot(np.log10(1./np.sqrt(sigma_sq[i,:])), np.log10(f_sigma[i,:]), 
                c[i]+'-', linewidth=2, label='$z = %3.2f$'%z[i])

    #data = np.loadtxt('./data/Tinker_hmf/mVector_PLANCK-SMT.txt')
    #ax.plot(np.log10(1./data[:,1]), np.log10(data[:,4]), 'k--', linewidth=2)

    ax.legend()
    #ax.semilogy()
    ax.set_xlabel(r'$\lg(1/\sigma)$')
    ax.set_ylabel(r'$\lg(f(\sigma))$')

    plt.savefig(output + 'fsigma.eps', formate='eps')

    #plt.show()

def plot_fitting_ABC(kh, A, B, C):

    kh_fitting = np.array([0.001, 0.01, 0.05, 0.07, 0.1, 0.5])
    A_fitting = np.array([0.0, 0.132, 0.613, 0.733, 0.786, 0.813])
    B_fitting = np.array([0.0, 1.620, 5.590, 6.000, 5.090, 0.803])
    C_fitting = np.array([0.0, 7.130, 21.13, 21.45, 15.50, -0.844])

    fig = plt.figure(figsize=(6,6))
    ax0  = fig.add_axes([0.14, 0.68, 0.8, 0.25])
    ax1  = fig.add_axes([0.14, 0.40, 0.8, 0.25])
    ax2  = fig.add_axes([0.14, 0.12, 0.8, 0.25])

    ax0.plot(kh, A, 'r-')
    ax1.plot(kh, B, 'r-')
    ax2.plot(kh, C, 'r-')

    ax0.plot(kh_fitting, A_fitting, 'ro')
    ax1.plot(kh_fitting, B_fitting, 'ro')
    ax2.plot(kh_fitting, C_fitting, 'ro')

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xlabel(r'$k\, [h/{\rm Mpc}]$')

    ax0.set_ylabel('$A(k)$')
    ax1.set_ylabel('$B(k)$')
    ax2.set_ylabel('$C(k)$')

def plot_fitting_f(kh, f, fnu_list, redshifts):
    fig = plt.figure(figsize=(6,4))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.80])
    c = 'rbgck'
    l = ['-', '--', ':', '-.']

    for i, fnu in enumerate(fnu_list):
        for j, z in enumerate(redshifts):
            if j == 0:
                ax.plot(kh, f[i, j], c[i]+l[j], label=r'$f_{\nu} = %3.2f$'%fnu, 
                        linewidth=2)
            else:
                ax.plot(kh, f[i, j], c[i]+l[j],  linewidth=2)

    ax.semilogx()
    ax.legend()
    ax.set_xlabel(r'$k\, [h/{\rm Mpc}]$')
    ax.set_ylabel(r'Growth Rate $f(k)$')

def plot_fitting(chisq_path, chisq_name):

    chisq_result = h5py.File(chisq_path + chisq_name, 'r')
    chisq = chisq_result['chisq'][:]
    x = chisq_result['x'][:] * 1.e4
    y = chisq_result['y'][:]

    like  = np.exp(-0.5*chisq)
    like /= like.max()

    n1, n2 = like.shape

    max_index = np.argmax(like)

    print max_index
    
    max_index0 = max_index/n2
    print max_index0
    max_index1 = max_index-max_index0*n2
    print max_index1
    y0 = y[max_index0]
    x0 = x[max_index1]


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0.15, 0.14, 0.8, 0.8])
    #cax= fig.add_axes([0.83, 0.21, 0.03, 0.61])

    X, Y = np.meshgrid(x, y)
    ax.pcolormesh(X, Y, like, cmap="Greens")
    ax.contour(X, Y, chisq, (chisq.min(), chisq.min()+2.3, chisq.min()+6.17), 
            linestyles='solid', linewidth=2, colors='r')

    ax.vlines(x0, Y.min(), Y.max(), 'k', linestyles='dashed', linewidths=2)
    ax.hlines(y0, X.min(), X.max(), 'k', linestyles='dashed', linewidths=2)

    ax.set_ylabel(r'$\Sigma m_{\nu}\, [{\rm eV}]$')
    ax.set_xlabel(r'$\bar{\tau} \times 10^{-4}$')

    plt.savefig('../png/'+ result_name.replace('.h5', '.png'), formate='png')

def plot_v_mnu(data_path, data_name):

    results = h5py.File(data_path + data_name, 'r')
    vij = results['vij'][:]
    rh  = results['rh'][:]
    zs  = results['zs'][:]
    nu  = results['nu'][:]
    print vij.shape
    print nu

    for i in range(vij.shape[1]):
        fig = plt.figure(figsize=(6,4))
        ax  = fig.add_axes([0.14, 0.47, 0.80, 0.50])
        ax2 = fig.add_axes([0.14, 0.14, 0.80, 0.29])
        c = 'rbgck'
        for j in range(vij.shape[0]):
            ax.plot(rh, -vij[j, i, :]/100., c[j]+'-', linewidth=2, 
                    label=r'$\Sigma m_{\nu} = ' + '%3.2f'%nu[j] + r'\,{\rm [eV]}$')
            ax2.plot(rh, vij[j, i, :]/vij[0, i, :], 
                    c[j]+'-', linewidth=2,)
                    #label=r'$\Sigma m_{\nu} = ' + '%3.2f'%nu[j] + r'\,{\rm [eV]}$')
        ax.legend(title='$z = %3.2f$'%zs[i])
        #ax.set_xlabel(r'$r\,[{\rm Mpc}/h]$')
        ax.set_xticklabels([])
        ax.set_ylabel(r'$-v_{ij}(r)\, [100{\rm km/s}]$')

        ax2.set_ylabel(r'$v_{ij}/v_{ij}^{\Sigma m_{\nu}=0}$')
        ax2.set_xlabel(r'$r\,[{\rm Mpc}/h]$')

        plt.savefig(data_path + 'eps/' + data_name.replace('.h5', '_z%3.2f.eps'%zs[i]), 
                formate='eps')

def plot_stellarmass_hist(data_path_list, label_list):

    c = 'rbgckm'
    fig = plt.figure(figsize=(5,4))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.8])

    #bins = np.linspace(9, 12, 50)
    bins = np.logspace(9.5, 12.5, 50)
    for i in range(len(data_path_list)):
        data_path = data_path_list[i]

        data = np.genfromtxt(data_path, names=True)
        data_n = data.shape[0]

        if i < 4:
            cgc = data['CGC'] >= 1

            has_mass = data['MC_MATCH'] != 0

            cgc_has_mass = np.logical_and(cgc, has_mass)
        if i == 5:
            cgc_has_mass = data['COLOR'] > np.percentile(data['COLOR'], 20)
        else:
            cgc_has_mass = np.ones(data_n).astype('bool')

        data_cgc = data[cgc_has_mass]

        hist, bins = np.histogram(10.**data_cgc['LOGMASS'], bins=bins)

        ax.plot(bins[:-1], hist * 1.e-4, c[i]+'-', drawstyle='steps-post', 
                label=label_list[i])

    ax.set_xlabel(r'$M_{\rm stellar}\, [h^{-1}M_{\odot}]$')
    ax.set_ylabel(r'${\rm Galaxy\, Number}\, [\times 10^4]$')
    ax.semilogx()
    ax.legend()


def plot_stellarmass(data_path):

    data = np.genfromtxt(data_path, names=True)
    data_n = data.shape[0]

    cgc = data['CGC'] >= 1

    has_mass = data['MC_MATCH'] == 1

    has_mass2 = data['MC_MATCH'] == 2

    data1 = data[has_mass]
    data2 = data[has_mass2]

    data1_cgc = data[np.logical_and(has_mass, cgc)]
    data2_cgc = data[np.logical_and(has_mass2, cgc)]

    print " Total galaxy number %d"%data.shape[0]
    print " M_Stellar from BOSS %d"%data1.shape[0]
    print "M_Stellar from GSWLC %d"%data2.shape[0]

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.8])

    ax.scatter(data1['Z'], data1['LOGMASS'] + np.log10(0.7**2), 
            s=5, c='k', marker='o', edgecolors='none', 
            label='BOSS')
    ax.scatter(data2['Z'], data2['LOGMASS'] + np.log10(0.7**2), 
            s=5, c='0.5', marker='o', edgecolors='none',
            label='GSWLC')
    ax.scatter(data1_cgc['Z'], data1_cgc['LOGMASS'] + np.log10(0.7**2), 
            s=5, c='r', marker='o', edgecolors='none', 
            label='BOSS CGC1')
    ax.scatter(data2_cgc['Z'], data2_cgc['LOGMASS'] + np.log10(0.7**2), 
            s=5, c='g', marker='o', edgecolors='none',
            label='GSWLC CGC1')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(7.9, 12.1)
    ax.set_ylabel(r'$\lg(M_{\rm stellar}/h^{-2}M_{\odot})$')
    ax.set_xlabel(r'z')
    ax.legend()

def plot_halomass():

    stellar_mass = np.logspace(9., 12., 100.)
    print stellar_mass

    halo_mass = models.stellarM2haloM(stellar_mass)

    print halo_mass

    fig = plt.figure(figsize=(6,6))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.8])

    ax.plot(stellar_mass, halo_mass, 'r-', linewidth=1.5)

    ax.set_xlabel('Stellar Mass $[M_\odot]$')
    ax.set_ylabel('Halo Mass $[M_\odot]$')
    ax.loglog()

#def plot_halomass_hist(data_path_list, label_list):
#
#    c = 'rbgckm'
#    fig = plt.figure(figsize=(5,4))
#    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.8])
#
#    #bins = np.linspace(9, 12, 50)
#    bins = np.logspace(10.5, 16.5, 50)
#    for i in range(len(data_path_list)):
#        data_path = data_path_list[i]
#
#        data = np.genfromtxt(data_path, names=True)
#        data_n = data.shape[0]
#
#        if i < 4:
#            cgc = data['CGC'] >= 1
#
#            has_mass = data['MC_MATCH'] != 0
#
#            cgc_has_mass = np.logical_and(cgc, has_mass)
#        #if i == 5:
#        #    cgc_has_mass = data['COLOR'] > np.percentile(data['COLOR'], 20)
#        else:
#            cgc_has_mass = np.ones(data_n).astype('bool')
#
#        data_cgc = data[cgc_has_mass]
#
#        halo_mass = models.stellarM2haloM((10.**data_cgc['LOGMASS'])/0.6774)
#        halo_mass *= 0.6774
#        hist, bins = np.histogram(halo_mass, bins=bins)
#
#        ax.plot(bins[:-1], hist/1.e4, c[i]+'-', drawstyle='steps-post', 
#                label=label_list[i])
#
#    #ax.set_xlabel(r'$\lg(M_{\rm halo}/h^{-2}M_{\odot})$')
#    ax.set_xlabel(r'$M_{\rm halo}\, [h^{-1}M_{\odot}]$')
#    ax.set_ylabel(r'${\rm Galaxy\, Number}\, [\times 10^4]$')
#    ax.semilogx()
#    ax.legend()
#
def plot_halomass_hist(data_path_list, label_list, H0=67.74):

    color = 'krbgcm'
    fig = plt.figure(figsize=(5,4))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.8])

    #bins = np.linspace(10.5, 16.5, 50)
    bins = np.logspace(10.5, 16.5, 50)
    ymax = 0.
    for i in range(len(data_path_list)):
        data_path = data_path_list[i]

        data = np.genfromtxt(data_path, names=True)
        data_n = data.shape[0]

        z_sel = np.logical_and(data['Z'] > 0.01, data['Z'] < 0.8)

        if 'CGC' in data.dtype.names:
            cgc = data['CGC'] >= 0
            if 'MC_MATCH' in data.dtype.names:
                has_mass = data['MC_MATCH'] != 0
                cgc_has_mass = np.logical_and(cgc, has_mass)
            else:
                cgc_has_mass = cgc
        else:
            cgc_has_mass = np.ones(data_n).astype('bool')

        #if 'IO' in data.dtype.names:
        #    cgc_has_mass = np.logical_and(cgc_has_mass, data['IO']==1)

        #if 'NMEM' in data.dtype.names:
        #    cgc_has_mass = np.logical_and(cgc_has_mass, data['NMEM'] > 1)

        data_cgc = data[np.logical_and(cgc_has_mass, z_sel)]

        if 'LOGHALOMASS' in data_cgc.dtype.names:
            halo_mass = 10.**data_cgc['LOGHALOMASS']
        else:
            halo_mass  = models.stellarM2haloM((10.**data_cgc['LOGMASS'])/(H0/100.))
            halo_mass *=  (H0/100.)


        z   = data_cgc['Z']

        median_mass = np.median(halo_mass)
        print '.'*50
        print data_path
        print halo_mass.shape
        print np.log10(median_mass)
        print halo_mass[halo_mass>median_mass].shape
        print halo_mass[halo_mass==median_mass].shape
        print halo_mass[halo_mass<median_mass].shape

        hist, bins = np.histogram(halo_mass, bins=bins)

        ax.plot(bins[:-1], hist/1.e4, color[i]+'-', linewidth=2, drawstyle='steps-post', 
                label=label_list[i])

        ymax = max(ymax, hist.max()/1.e4)

        #ax.vlines(np.log10(median_mass), ymin=0., ymax=10., colors=color[i], 
        #        linestyles='--')

    ax.semilogx()
    ax.set_xlabel(r'$M_{\rm halo}\, [h^{-1}M_{\odot}]$')
    #ax.set_ylabel(r'${\rm Galaxy\, Number}\, [\times 10^4]$')
    ax.set_ylabel(r'Sample Amount $\times10^{4}$')
    ax.set_ylim(0, ymax*(1.01))
    ax.legend()


def plot_rvir_hist(data_path_list, label_list, rho_crit = 2.775e11, ap_list=None):

    c = cosmo.Cosmology()
    c = c.init_physical(ombh2=0.02230, omch2=0.1188, H0=67.74, omkh2=0.00037)

    color = 'rbgckm'
    fig = plt.figure(figsize=(5,4))
    ax  = fig.add_axes([0.14, 0.14, 0.8, 0.8])

    bins = np.linspace(1, 15, 50)
    for i in range(len(data_path_list)):
        data_path = data_path_list[i]

        data = np.genfromtxt(data_path, names=True)
        data_n = data.shape[0]

        #if i < 4:
        #    cgc = data['CGC'] >= 1

        #    has_mass = data['MC_MATCH'] != 0

        #    cgc_has_mass = np.logical_and(cgc, has_mass)
        #if i == 5:
        #    cgc_has_mass = data['COLOR'] < np.percentile(data['COLOR'], 80)
        #else:
        #    cgc_has_mass = np.ones(data_n).astype('bool')

        z_sel = np.logical_and(data['Z'] > 0.01, data['Z'] < 0.4)

        if 'CGC' in data.dtype.names:
            cgc = data['CGC'] >= 1
            if 'MC_MATCH' in data.dtype.names:
                has_mass = data['MC_MATCH'] != 0
                cgc_has_mass = np.logical_and(cgc, has_mass)
            else:
                cgc_has_mass = cgc
        else:
            cgc_has_mass = np.ones(data_n).astype('bool')

        data_cgc = data[np.logical_and(cgc_has_mass, z_sel)]

        if 'LOGHALOMASS' in data_cgc.dtype.names:
            halo_mass = 10.**data_cgc['LOGHALOMASS']
        else:
            halo_mass  = models.stellarM2haloM((10.**data_cgc['LOGMASS'])/(c.H0/100.))
            halo_mass *=  (c.H0/100.)

        z   = data_cgc['Z']
        Ez  = c.H(z) / c.H()
        omg = c.omega_m * (1. + z) ** 3. / Ez**2
        Delta_c = 18. * np.pi**2. + 82. * (omg - 1.) - 39. * (omg - 1) ** 2.
        #rvir = (3. * halo_mass / 4. / np.pi / Delta_c / rho_crit) ** (1./3.)
        rvir = (3. * halo_mass / 4. / np.pi / 200. / rho_crit) ** (1./3.)
        if "DC" not in data_cgc.dtype.names:
            dC = c.comoving_distance(z)
        else:
            dC = data_cgc['DC']
        dA =  dC / (1. + z)

        theta = rvir / dA * 180./np.pi * 60.
        print theta.max(), theta.min()

        hist, bins = np.histogram(theta, bins=bins)

        ax.plot(bins[:-1], hist/1.e4, color[i]+'-', drawstyle='steps-post', 
                label=label_list[i])
        if ap_list is not None:
            if ap_list[i] != 0:
                ax.vlines(ap_list[i], ymin=0, ymax=10, colors=color[i], 
                        linewidths=2, linestyles='dotted')

    ax.set_xlabel(r'$\theta_{200}\, [{\rm arcmin}]$')
    ax.set_ylabel(r'${\rm Galaxy\, Number}\, [\times 10^4]$')
    ax.set_ylim(0, 6.5)
    ax.legend()

def plot_mcmc(mcmc_froot, mcmc_fnames, plot_labels, 
        planck_root=None, planck_name=None ):

    plot_froots = []

    for mcmc_fname in mcmc_fnames:

        plot_froot = mcmc_froot + mcmc_fname + '_plot/data/'
        if not os.path.exists(plot_froot):
            os.makedirs(plot_froot)

        num_bin_2d = 100
        num_bin_1d = 48

        get_dist.get_dist(mcmc_froot, mcmc_fname, ignore_frac=0.5, num_bin_2d=num_bin_2d,
                smear_factor = 3., num_bin_1d = num_bin_1d,
                ext='.h5', data_plot = plot_froot)

        plot_froots.append(mcmc_froot + mcmc_fname + '_plot/data/' + mcmc_fname,)

    if planck_root is not None:

        plot_froot = planck_root + planck_name + '_plot/data/'
        if not os.path.exists(plot_froot):
            os.makedirs(plot_froot)

        num_bin_2d = 100
        num_bin_1d = 48

        get_dist.get_dist(planck_root, planck_name, param_x=['w', 'wa'],
                param_y=['w', 'wa'], ignore_frac=0.5, num_bin_2d=num_bin_2d,
                smear_factor = 2., num_bin_1d = num_bin_1d,
                ext='.txt', data_plot = plot_froot)

        plot_froots.append(planck_root + planck_name + '_plot/data/' + planck_name,)
        plot_labels.append('Planck')

    plot_triangle.plot_triangle('', plot_froots, ['tau', 'w', 'wa'], plot_labels)
    #plot_2d.plot_contours('', plot_froots, x_param='w', y_param='wa', labels=plot_labels)
    #plot_1d.plot_profile('', plot_froots, x_param='tau', labels=plot_labels)




if __name__=="__main__":

    plot_z = plot_m = plot_result = plot_chisq = plot_bias = plot_fit = False
    plot_v = plot_smass = plot_hmass =  mcmc = False

    #plot_z = True
    #plot_m = True
    #plot_result = True
    #plot_chisq = True
    #plot_bias = True
    #plot_fit = True
    #plot_v = True
    #plot_smass = True
    #plot_hmass = True
    mcmc = True

    if mcmc:

        mcmc_froot = '/data/ycli/ksz/'
        mcmc_fnames = [
                #'pkSZ_result_tSZcln_DR13GPCM_50_AP7arcm_f1.41_sigz0.001tau_w_wa_MHchains',
                #'pkSZ_result_tSZcln_DR12LOWZNwMASS_CGC1_50_AP7arcm_f1.41_sigz0.001tau_w_wa_chains',
                #'pkSZ_result_tSZcln_DR12LOWZSwMASS_CGC1_50_AP7arcm_f1.41_sigz0.001tau_w_wa_chains',

                #'pkSZ_result_tSZcln_DR13GPCM_50_AP7arcm_f1.41_sigz0.001tau_w_chains',
                #'pkSZ_result_tSZcln_DR12LOWZNwMASS_CGC1_50_AP7arcm_f1.41_sigz0.001tau_w_chains',
                #'pkSZ_result_tSZcln_DR12LOWZSwMASS_CGC1_50_AP7arcm_f1.41_sigz0.001tau_w_chains',

                #'pkSZ_result_tSZcln_DR13GPCM_50_AP7arcm_f1.41_sigz0.001tau_chains',
                #'pkSZ_result_tSZcln_DR12LOWZNwMASS_CGC1_50_AP7arcm_f1.41_sigz0.001tau_chains',
                #'pkSZ_result_tSZcln_DR12LOWZSwMASS_CGC1_50_AP7arcm_f1.41_sigz0.001tau_chains',
                #'pkSZ_result_newbin_tSZcln_DR13GPCM_JK_50_AP8arcm_f1.41_sigz0.001tau_w_chains',
                'pkSZ_result_newbin_tSZcln_DR13GPCM_JK_50_AP8arcm_f1.41_sigz0.001tau_w_wa_chains',
                'pkSZ_result_newbin_tSZcln_DR12LOWZNwMASS_CGC1_JK_50_AP8arcm_f1.41_sigz0.001tau_w_wa_chains',

                ]
        plot_labels = [
                #'DR13 Group Catalog',
                #'LowZ North',
                #'LowZ Sorth',

                #'DR13 Group Catalog',
                #'LowZ North',
                #'LowZ Sorth',

                #'DR13 Group Catalog',
                #'LowZ North',
                #'LowZ Sorth',
                'DR13 Group Catalog',
                'LowZ North',
                ]

        #planck_root = '/data/ycli/planck_mcmc/base_w_wa/planck_lowl_lowLike_BAO/'
        #planck_name = 'base_w_wa_planck_lowl_lowLike_BAO'
        #planck_root = '/data/ycli/planck_mcmc/base_w_wa/planck_lowl_lowLike_highL_BAO/'
        #planck_name = 'base_w_wa_planck_lowl_lowLike_highL_BAO_post_HST'
        #planck_root = '/data/ycli/planck_mcmc/base_w_wa/planck_lowl_lowLike_SNLS/'
        #planck_name = 'base_w_wa_planck_lowl_lowLike_SNLS'
        planck_root = None
        planck_name = None
        plot_mcmc(mcmc_froot, mcmc_fnames, plot_labels, planck_root, planck_name)

    if plot_hmass:

        #plot_halomass()
        data_path_list = [
                '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_TOT_wMASS.dat',
                '/data/ycli/dr12/galaxy_DR12v5_LOWZ_South_TOT_wMASS.dat',
                '/data/ycli/dr12/galaxy_DR12v5_CMASS_North_TOT_wMASS.dat',
                '/data/ycli/dr12/galaxy_DR12v5_CMASS_South_TOT_wMASS.dat',
                '/data/ycli/cgc/CGC_wMASS.dat',
                '/data/ycli/6df/6dFGS_2MASS_RA_DEC_Z_J_K_bJ_rF_GOOD.cat'
                ]
        label_list = [
                'DR12 LOWZ North CGC1',
                'DR12 LOWZ South CGC1',
                'DR12 CMASS North CGC1',
                'DR12 CMASS South CGC1',
                'DR7 CGC',
                '6dF RT80'
                ]
        ap_list = [
                7.,
                0.,
                0.,
                0.,
                8.,
                11.,
                ]
        #plot_halomass_hist(data_path_list, label_list)
        plot_rvir_hist(data_path_list, label_list, rho_crit = 2.775e11, ap_list=ap_list)

    if plot_smass:

        data_path_list = [
                '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_TOT_wMASS.dat',
                '/data/ycli/dr12/galaxy_DR12v5_LOWZ_South_TOT_wMASS.dat',
                '/data/ycli/dr12/galaxy_DR12v5_CMASS_North_TOT_wMASS.dat',
                '/data/ycli/dr12/galaxy_DR12v5_CMASS_South_TOT_wMASS.dat',
                '/data/ycli/cgc/CGC_wMASS.dat',
                '/data/ycli/6df/6dFGS_2MASS_RA_DEC_Z_J_K_bJ_rF_GOOD.cat'
                ]
        label_list = [
                'DR12 LOWZ North CGC1',
                'DR12 LOWZ South CGC1',
                'DR12 CMASS North CGC1',
                'DR12 CMASS South CGC1',
                'DR7 CGC',
                '6dF RT80'
                ]
        #plot_stellarmass(data_path)
        plot_stellarmass_hist(data_path_list, label_list)

    if plot_v:
        data_path = './data/'
        data_name = 'new_vij_mnu_Mmin14_nowindow_z0.05.h5'
        plot_v_mnu(data_path, data_name)

    if plot_fit:
        result_path = '/data/ycli/ksz/'
        result_name = 'pkSZ_result_tSZcln_DR12LOWZNCGC_50_AP5arcm_chisq.h5'
        plot_fitting(result_path, result_name)


    if plot_bias:
        data_path = './data/'
        data_file = [
                '%s_mu0.txt',
                '%s_mu0.06.txt', 
                '%s_mu0.6.txt',
                ]
        label_list = [
                r'$\Sigma m_{\nu} = 0 {\rm eV}$',
                r'$\Sigma m_{\nu} = 0.06 {\rm eV}$',
                r'$\Sigma m_{\nu} = 0.6 {\rm eV}$',
                ]
        plot_bias_multi(data_path, data_file, label_list, z_indx=0)

    #check_map()
    if plot_z:
        data_list = [
                '/data/ycli/cgc/galcat_CGC.dat',
                #'/data/ycli/gama/galcat_GAMA.dat',
                '/data/ycli/dr12/galcat_DR12LOWZNwMASS.dat',
                '/data/ycli/dr12/galcat_DR12LOWZSwMASS.dat',
                #'/data/ycli/dr12/galcat_DR12CMASNwMASS.dat',
                #'/data/ycli/dr12/galcat_DR12CMASSwMASS.dat',

                #'/data/ycli/6df/galcat_6dFGSDR3.dat',
                #'/data/ycli/group_catalog/galcat_6dFGPCM.dat',
                '/data/ycli/group_catalog/galcat_DR13GPCM.dat',
                ]
        label_list = [
                'DR7 CGC',
                #'GAMA',
                'DR12 LOWZ North LSS',
                'DR12 LOWZ South LSS',
                #'DR12 CMASS North LSS',
                #'DR12 CMASS South LSS',

                #'6dF',
                #'6dF group',
                'dr13 group',
                ]
        z_list = ['Z', 'Z', 'Z', 'Z', 'Z', 'Z'] #, 'Z_TONRY']
        #output = '../png/z_hist_DR12_North.eps'
        output = '../png/z_hist_all.eps'
        #output = '../png/z_hist_DR12_South.eps'
        plot_z_hist(data_list, label_list, z_list, output=output)

    if plot_m:
        data_list = [
                #'/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_RaDecZ.dat',
                #'/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_RaDecZ_CGC.dat',
                #'/data/ycli/dr12/galaxy_DR12v5_CMASS_North_RaDecZ.dat',
                #'/data/ycli/dr12/galaxy_DR12v5_CMASS_North_RaDecZ_CGC.dat',
                '/data/ycli/dr12/galaxy_DR12v5_LOWZ_South_RaDecZ.dat',
                '/data/ycli/dr12/galaxy_DR12v5_LOWZ_South_RaDecZ_CGC.dat',
                '/data/ycli/dr12/galaxy_DR12v5_CMASS_South_RaDecZ.dat',
                '/data/ycli/dr12/galaxy_DR12v5_CMASS_South_RaDecZ_CGC.dat',
                ]
        label_list = [
                #'DR12 LOWZ North LSS',
                #'DR12 LOWZ North LSS CGC',
                #'DR12 CMASS North LSS',
                #'DR12 CMASS North LSS CGC',
                'DR12 LOWZ South LSS',
                'DR12 LOWZ South LSS CGC',
                'DR12 CMASS South LSS',
                'DR12 CMASS South LSS CGC',
                ]
        m_list = ['MODELFLUX_r', 'MODELFLUX_r', 'MODELFLUX_r', 'MODELFLUX_r']
        #output = '../png/m_hist_DR12_North.eps'
        output = '../png/m_hist_DR12_South.eps'
        plot_m_hist(data_list, label_list, m_list, output=output)

    if plot_result:

        result_path = '/data/ycli/ksz/'
        output_path = '../png/'
        result_name_list = [
                #'pkSZ_result_ko_HFI217_CGC_50_AP8arcm.h5',
                #'pkSZ_result_ko_HSEVEM_CGC_50_AP8arcm.h5',
                'pkSZ_result_ko_tSZcln_CGC_50_AP8arcm.h5',
                #'pkSZ_result_shuffle_tSZcln_CGC_50_AP8arcm.h5',

                'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC1_50_AP8arcm.h5',
                #'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC2_50_AP8arcm.h5',
                'pkSZ_result_shuffle_tSZcln_6dFGSDR3_50_AP8arcm.h5',

                #'pkSZ_result_tSZcln_DR12LOWZNCGC_50_AP8arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC_mlim19.5_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC4_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC2_mlim19.5_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZNB50CG_50_AP5arcm.h5'

                #'pkSZ_result_tSZcln_DR12LOWZSCGC_50_AP8arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZSCGC_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZSCGC_mlim19.5_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZSCGC4_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12LOWZSCGC2_mlim19.5_50_AP5arcm.h5',

                #'pkSZ_result_tSZcln_DR12CMASNCGC_20_AP8arcm.h5',
                #'pkSZ_result_tSZcln_DR12CMASNCGC_mlim20.5_20_AP8arcm.h5',
                #'pkSZ_result_tSZcln_DR12CMASNCGC4_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12CMASNCGC2_mlim20.5_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12CMASNB50CG_50_AP5arcm.h5',

                #'pkSZ_result_tSZcln_DR12CMASSCGC_40_AP8arcm.h5',
                #'pkSZ_result_tSZcln_DR12CMASSCGC4_50_AP5arcm.h5',
                #'pkSZ_result_tSZcln_DR12CMASSB50CG_50_AP5arcm.h5',
                ]
        label_list = [
                #'HFI 217 GHz map',
                #'SEVEM map',
                'DR7 CGC',
                #'DR7 CGC, Shuffle',
                '6dF CGC, Shuffle',
                #'6dF CGC 2, Shuffle',
                '6dF, Shuffle',

                #'DR12 LOWZ North CGC 8 arcm',
                #'DR12 LOWZ North CGC 5 arcm',
                #'DR12 LOWZ North CGC 5 arcm $r<19.5$',
                #'DR12 LOWZ North CGC4 5 arcm',
                #'DR12 LOWZ North CGC2 5 arcm $r<19.5$',
                #'DR12 LOWZ North CGC B50  5 arcm',

                #'DR12 LOWZ South CGC 8 arcm',
                #'DR12 LOWZ South CGC 5 arcm',
                #'DR12 LOWZ South CGC 5 arcm $r<19.5$',
                #'DR12 LOWZ South CGC4 5 arcm',
                #'DR12 LOWZ South CGC2 5 arcm $r<19.5$',

                #'DR12 CMASS North CGC',
                #'DR12 CMASS North CGC r<20.5',
                #'DR12 CMASS North CGC 4',
                #'DR12 CMASS North CGC 2 r<20.5',
                #'DR12 CMASS North CGC B50',

                #'DR12 CMASS South CGC',
                #'DR12 CMASS South CGC 4',
                #'DR12 CMASS South CGC B50',
                ]
        text = ''
        output_name = 'plot_6dFGSDR3'
        #output_name = 'plot_DR12_LOWZNCGC'
        #output_name = 'plot_DR12_LOWZSCGC'
        #output_name = 'plot_DR12_CMASCGC'

        #check_TAP_bar(result_path, result_name_list,  label_list=label_list,
        #        output_path=output_path, output_name=output_name, text=text)
        #plot_cov(result_path, result_name_list, label_list=label_list,
        #        output_path=output_path, text=text)
        plot_pkSZ(result_path, result_name_list, label_list=label_list,
                output_path=output_path, output_name=output_name,text=text)
        plot_error(result_path, result_name_list, label_list=label_list,
                output_path=output_path, output_name=output_name,text=text)

    if plot_chisq:
        result_path = '/data/ycli/ksz/'
        output_path = '../png/'
        #ap_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        #ap_list = [7, 8, 9, 10, 11, 12]
        ap_list = [11, 10] #, 9, 8, 7, 6]
        result_name_list = [
                'pkSZ_result_ko_tSZcln_CGC_50',
                #'pkSZ_result_tSZcln_DR12LOWZSCGC_50',
                'pkSZ_result_tSZcln_DR12LOWZNCGC_50',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC_zlim0.00to0.20_50',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC_zlim0.20to0.30_50',
                #'pkSZ_result_tSZcln_DR12LOWZNCGC_zlim0.30to0.45_50',
                'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC1_50',
                ]
        label_list = [
                'DR7 CGC',
                #'DR12 LOWZ South CGC',
                'DR12 LOWZ North CGC',
                #'DR12 LOWZ North CGC, $z\in[0.00, 0.20)$',
                #'DR12 LOWZ North CGC, $z\in[0.20, 0.30)$',
                #'DR12 LOWZ North CGC, $z\in[0.30, 0.45)$',
                '6dF CGC'
                ]
        output_name = 'plot_6dF'
        #plot_chisq_vs_ap(result_path, result_name_list, ap_list, 
        #        label_list=label_list, output_path=output_path, 
        #        output_name=output_name)

        for ap in ap_list:
            result_name_list = [
                    #'pkSZ_result_tSZcln_DR12LOWZSCGC_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_tSZcln_DR12LOWZNCGC_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_tSZcln_DR12LOWZNCGC_zlim0.00to0.20_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_tSZcln_DR12LOWZNCGC_zlim0.20to0.30_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_tSZcln_DR12LOWZNCGC_zlim0.30to0.45_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC1_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSK70CGC1_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSK70_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT50_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT80_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT20_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT80OLD_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT80TEST_50_AP%darcm.h5'%ap,
                    'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT80TEST_50_AP%darcm.h5'%ap,
                    'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT80TEST_50_AP%darcm_f2.00.h5'%ap,
                    'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC1TEST_50_AP%darcm.h5'%ap,
                    'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC1TEST_50_AP%darcm_f2.00.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3CGC1TEST_50_AP%darcm_f1.10.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3MT80TEST_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3MT50TEST_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3BT50_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3BT20_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3BT80_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3RT80MT60_50_AP%darcm.h5'%ap,
                    #'pkSZ_result_shuffle_tSZcln_6dFGSDR3MT50_50_AP%darcm.h5'%ap,
                    ]
            label_list = [
                    #'DR12 LOWZ South CGC AP= $%d$ arcmin'%ap,
                    #'DR12 LOWZ North CGC AP= $%d$ arcmin'%ap,
                    #'DR12 LOWZ North CGC, $z\in[0.00, 0.20)$',
                    #'DR12 LOWZ North CGC, $z\in[0.20, 0.30)$',
                    #'DR12 LOWZ North CGC, $z\in[0.30, 0.45)$',
                    #'6dF CGC AP = $%d$ arcmin'%ap,
                    #'6dF K70 CGC AP = $%d$ arcmin'%ap,
                    #'6dF K70 AP = $%d$ arcmin'%ap,
                    #'6dF RT50 AP = $%d$ arcmin'%ap,
                    #'6dF RT80 zflag AP = $%d$ arcmin'%ap,
                    #'6dF RT80 AP = $%d$ arcmin'%ap,
                    #'6dF RT80 old AP = $%d$ arcmin'%ap,
                    #r'6dF Red Top $80\%$ test '+'AP = $%d$ arcmin'%ap,
                    r'6dF Red Top $80\%$ test '+'AP = $%d$ arcmin'%ap,
                    r'6dF Red Top $80\%$ test '+'AP = $%d$ arcmin f = 2.0'%ap,
                    '6dF CGC1 test AP = $%d$ arcmin'%ap,
                    '6dF CGC1 test AP = $%d$ arcmin f = 2.0'%ap,
                    #'6dF CGC1 test AP = $%d$ arcmin f = 1.1'%ap,
                    #'6dF MT80 test AP = $%d$ arcmin'%ap,
                    #'6dF MT50 test AP = $%d$ arcmin'%ap,
                    #'6dF BT50 AP = $%d$ arcmin'%ap,
                    #'6dF BT20 AP = $%d$ arcmin'%ap,
                    #'6dF BT80 AP = $%d$ arcmin'%ap,
                    #'6dF RT80 MT60 AP = $%d$ arcmin'%ap,
                    #'6dF MT50 AP = $%d$ arcmin'%ap,
                    ]
            #output_name = 'plot_DR12LOWZ_AP%d'%ap
            output_name = 'plot_6dF_AP%d'%ap
            text=''
            plot_pkSZ(result_path, result_name_list, label_list=label_list,
                    output_path=output_path, output_name=output_name,text=text,
                    ymax=0.12, ymin=-0.3)
            plot_error(result_path, result_name_list, label_list=label_list,
                    output_path=output_path, output_name=output_name,text=text, 
                    ymax=0.2, ymin=0.0)

    plt.show()




