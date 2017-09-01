import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import models
import camb_wrap
import h5py
#from mpi4py import MPI
from emcee.utils import MPIPool
import time

import emcee
import sys
import os

_camb = camb_wrap.CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')

def lnlike(theta, x, y, yerr, ycovi, param, 
        cosm_param_idx=0, camb_run=False, T_CMB=2.7255, **kwargs):

    param_n = param.shape[0]
    names = '%s, ' * param_n
    names = names%tuple(param)
    p = np.core.records.fromarrays(theta[:,None], names=names)

    #camb = models.CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')
    #t1 = time.time()

    #tau_bar, mnu, ombh2, omch2 = theta
    #tau_bar, mnu, omch2 = theta
    #tau_bar, w = theta
    c = _camb
    factor = T_CMB * 1.e6 / 2.99e5

    for i in range(cosm_param_idx, param_n):
        key = param[i]
        c.params[key] = p[key][0]

    if camb_run:
        c.set_cosmology_params()
        c.set_darkenergy_params()
        c.get_results()

    vij, rh = models.mean_pairwise_v(c, rh=x, **kwargs)

    amp = p['tau'][0] * factor * 1.e-4

    if ycovi is not None:
        chisq = np.dot(np.dot((vij * amp - y), ycovi), (vij * amp - y).T)[0, 0]
    elif yerr is not None:
        chisq  = np.sum((vij * amp - y)**2. / yerr ** 2.)
    else:
        print "yerr or ycovi must be given!!"
        exit()
    if not np.isfinite(chisq):
        return -np.inf

    #print "[TIMING] One Step: %8.4f [s]"%(time.time() - t1)
    return -0.5 * chisq

def lnprior(theta, theta_min=None, theta_max=None):

    if theta_min is None: 
        print "params min not set"
        theta_min = theta
    if theta_max is None: 
        print "params max not set"
        theta_max = theta

    for i in range(len(theta)):
        if theta[i] > theta_max[i] or theta[i] < theta_min[i]:
            return -np.inf
    return 0.

def lnprob(theta, x, y, yerr, ycovi, param, theta_min=None, theta_max=None, **kwargs):

    lp = lnprior(theta, theta_min, theta_max)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, x, y, yerr, ycovi, param, **kwargs)

class MH_proposal(object):
    '''
    A Metropolis-Hastings proposal
    '''

    def __init__(self, cov, random = None):

        self.cov = cov
        if random is not None:
            self._random = random
        else:
            self._random = np.random.mtrand.RandomState()

    def __call__(self, X):
        (nw, npar) = X.shape
        mean = np.zeros(npar)
        return self._random.multivariate_normal(mean, self.cov, size=nw) + X

def read_param_dict(param_dict, camb_param=[]):

    camb_run = False
    cosm_param_idx = 0

    param_keys = param_dict.keys()

    comm_param = []
    cosm_param = []

    comm_param_ini = []
    cosm_param_ini = []

    comm_param_min = []
    cosm_param_min = []

    comm_param_max = []
    cosm_param_max = []

    comm_param_tex = []
    cosm_param_tex = []

    if 'w' in param_keys:
        cosm_param.append('w')
        cosm_param_ini.append(param_dict['w'][0])
        cosm_param_min.append(param_dict['w'][1])
        cosm_param_max.append(param_dict['w'][2])
        cosm_param_tex.append('w')

    if 'wa' in param_keys:
        cosm_param.append('wa')
        cosm_param_ini.append(param_dict['wa'][0])
        cosm_param_min.append(param_dict['wa'][1])
        cosm_param_max.append(param_dict['wa'][2])
        cosm_param_tex.append('wa')

    for key in param_keys:
        if key in ['w', 'wa']: continue
        if key in camb_param:
            cosm_param.append(key)
            cosm_param_ini.append(param_dict[key][0])
            cosm_param_min.append(param_dict[key][1])
            cosm_param_max.append(param_dict[key][2])
            cosm_param_tex.append(param_dict[key][3])
            camb_run = True
        else:
            comm_param.append(key)
            comm_param_ini.append(param_dict[key][0])
            comm_param_min.append(param_dict[key][1])
            comm_param_max.append(param_dict[key][2])
            comm_param_tex.append(param_dict[key][3])

    cosm_param_idx = len(comm_param)

    param     = np.array(comm_param + cosm_param)
    param_ini = np.array(comm_param_ini + cosm_param_ini)
    param_min = np.array(comm_param_min + cosm_param_min)
    param_max = np.array(comm_param_max + cosm_param_max)
    param_tex = np.array(comm_param_tex + cosm_param_tex)

    return param, param_ini, param_min, param_max, param_tex, cosm_param_idx, camb_run


def mcmc(result_path, result_name, param_dict, nwalkers=10, steps=2, 
        output='./mcmc_chains.h5', overwrite=False, covmat=None, 
        redshift=0.05, logmass_min=11., logmass_max=16.):

    pool = MPIPool(loadbalance=True, debug=False)
    rank = pool.rank
    comm = pool.comm

    # initializign CAMB power spectrum
    _camb.set_matter_power(redshifts=[redshift, ], kmax=3.e1)



    if rank == 0:
        result = h5py.File(result_path + result_name, 'r')

        pksz = result['pkSZ'][:]
        jk_sample = False
        if 'pkSZ_random' in result.keys():
            pksz_random = result['pkSZ_random'][:]
            jk_sample = False
        elif 'pkSZ_jk' in result.keys():
            pksz_random = result['pkSZ_jk'][:]
            jk_sample = True
        else:
            print "Need random samples"
            exit()
        pksz_bins = result['pkSZ_bins'][:]

        d_bins = pksz_bins[1:] - pksz_bins[:-1]
        pksz_bins = pksz_bins[:-1] + 0.5 * d_bins

        result.close()

        lin_scale = pksz_bins > 25.
        pksz_obs = pksz[lin_scale]

        pksz_err = None
        pksz_cov = np.cov(pksz_random, rowvar=False, bias=jk_sample)
        if jk_sample:
            spl_n = float(pksz_random.shape[0] - 1)
            bin_n = pksz_cov.shape[0]
            pksz_cov  *= spl_n
            pksz_covm  = np.linalg.inv(pksz_cov[:, lin_scale][lin_scale, :])
            pksz_covm *= (spl_n - bin_n) / spl_n
        else:
            pksz_covm = np.linalg.inv(pksz_cov[:, lin_scale][lin_scale, :])

        pksz_bins = pksz_bins[lin_scale]

    else:
        pksz_obs = None
        pksz_err = None
        pksz_bins = None
        pksz_covm = None

    pksz_obs = comm.bcast(pksz_obs, root=0)
    #pksz_err = comm.bcast(pksz_err, root=0)
    pksz_bins = comm.bcast(pksz_bins, root=0)
    pksz_covm = comm.bcast(pksz_covm, root=0)

    comm.barrier()

    if rank != 0:
        pool.wait()
        sys.exit(0)

    param, theta, theta_min, theta_max, param_tex, cosm_param_idx, camb_run\
            = read_param_dict(param_dict, camb_param=_camb.params.keys())

    paramnames = open(output.replace('.h5', '.paramnames'), 'w')
    for i in range(param.shape[0]):
        paramnames.write('%s\t\t'%param[i] + param_tex[i] + '\n')
    paramnames.close()

    # param = [cent, min, max]
    #tau_bar = [1., -1., 3.]
    #mnu     = [0.4, 0.2, 0.6]
    #ombh2   = [0.0221, 0.005, 0.1]
    #omch2   = [0.12, 0.001, 0.99]
    #w       = [-1, -3., 1.]
    #wa      = [0., -3., 3.]

    #theta     = param
    #theta_min = np.array([tau_bar[1], w[1]])
    #theta_max = np.array([tau_bar[2], w[2]])

    ndim = theta.shape[0]

    #threads = nwalkers
    threads = 1

    pos = [theta + 1.e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=threads,
            pool=pool, args=(pksz_bins, pksz_obs, pksz_err, pksz_covm, param),
            kwargs={ 'theta_min'      :theta_min, 
                     'theta_max'      :theta_max,
                     'cosm_param_idx' :cosm_param_idx, 
                     'camb_run'       :camb_run, 
                     'T_CMB'          :2.7255, 
                     'logmass_min'    :logmass_min, 
                     'logmass_max'    :logmass_max})

    # Run 100 steps as a burn-in.
    #pos, prob, state = sampler.run_mcmc(pos, 100)
    # Reset the chain to remove the burn-in samples.
    #sampler.reset()
    # Starting from the final position in the burn-in chain
    #sampler.run_mcmc(pos, steps, rstate0=state)


    #chain = np.zeros((size, ) + chain_local.shape )
    #comm.Gather(chain_local, chain, root=0)

    step_group = 100
    if step_group > steps: step_group = steps
    n_steps = steps / step_group
    state = None
    if rank == 0:
        if overwrite or not os.path.exists(output):
            mcmc_chains = h5py.File(output, 'w')
            #mcmc_chains['n_steps'] = n_steps
            mcmc_chains['n_steps'] = 0
            mcmc_chains['pos'] = pos
            if covmat is not None:
                mcmc_chains['covmat'] = np.zeros((ndim, ndim))
            #mcmc_chains['state'] = 0
            n_steps0 = 0
            mcmc_chains.close()
        else:
            mcmc_chains = h5py.File(output, 'a')
            n_steps0 = mcmc_chains['n_steps'][...]
            pos = mcmc_chains['pos'][...]
            if covmat is not None and n_steps0 != 0:
                covmat = mcmc_chains['covmat']
            #state = mcmc_chains['state'][...]
            #if state == 0: state = None
            mcmc_chains.close()

    for i in range(n_steps0, n_steps0 + n_steps):

        if rank == 0:
            t1 = time.time()

        if covmat is not None:
            mh_proposal = MH_proposal(covmat, sampler._random)
        else:
            mh_proposal = None
        #mh_proposal = None


        pos, prob, state = sampler.run_mcmc(pos, step_group, state, 
                mh_proposal=mh_proposal)

        if rank == 0:

            chain = sampler.chain
            chisq = sampler.lnprobability * (-2.)
            mcmc_chains = h5py.File(output, 'a')
            mcmc_chains['chains_%02d'%i] = chain
            mcmc_chains['chisqs_%02d'%i] = chisq
            mcmc_chains['n_steps'][...] = i + 1
            mcmc_chains['pos'][...] = pos
            #mcmc_chains['state'][...] = state
            #pksz_covm = np.cov(pksz_random[:, lin_scale], rowvar=False)
            if covmat is not None:
                covmat = np.cov(chain.reshape(-1, chain.shape[-1]), rowvar=False)
                mcmc_chains['covmat'][...] = covmat
            mcmc_chains.close()
            print "[TIMING] %3d x %4d Steps: %8.4f [s]"\
                    %(nwalkers, step_group, time.time() - t1)
            print "Mean acceptance fraction: ", np.mean(sampler.acceptance_fraction)

            sampler.reset()


    pool.close()

def analysis_mcmc(result_path, result_name, 
        planck_path=None, planck_name=None, planck_n=0):

    import corner

    fig=None

    if planck_path is not None:
        pchain = []
        for i in range(planck_n):
            pchain.append(np.loadtxt(data_path + data_name + '_%d.txt'%(i+1))[:, :9])

        pchain = np.concatenate(pchain, axis=0)
        weights = pchain[:,0]
        pchain = pchain[:, (6, 3)]

        fig = corner.corner(pchain, bins=50, weights=weights,
                verbose=True, smooth=2., smooth1d=2, color='r')

    mcmc_chains = h5py.File(result_path + result_name, 'r')
    n = mcmc_chains['n_steps'][...]
    print n
    chain_list = []
    chisq_list = []
    for i in range(10, n):
        chain = mcmc_chains['chains_%02d'%i][...]
        chisq = mcmc_chains['chisqs_%02d'%i][...]
        #chain = chain.reshape([-1, 4])
        chain_list.append(chain)
        chisq_list.append(chisq)

    chain = np.concatenate(chain_list, axis=1)
    chisq = np.concatenate(chisq_list, axis=1)
    print chain.shape

    acc = (chisq - np.roll(chisq, 1, axis=1) != 0)

    chain = chain.reshape([-1, chain.shape[-1]])
    acc = acc.flatten()
    chain = chain[acc, :]

    #chain = chain[:, 1:]

    labels = [
            r"$\bar{\tau}$", 
            #r"$\Sigma m_{\nu}$", 
            #r"$\Omega_{\rm b}h^2$",
            #r"$\Omega_{\rm c}h^2$", 
            #r"$\Omega_{\rm m}h^2$"
            r"$w$",
            r"$w_a$",
            ]
    corner.corner(chain, bins=50, smooth=2., smooth1d=2,
            verbose=True, fig=fig,
            labels=labels, color='k', max_n_ticks=3)
            #truths=[m_true, b_true, np.log(f_true)])

    plt.show()

def plot_planck_mcmc(data_path, data_name, n=8):

    import corner

    chain = []
    for i in range(n):
        chain.append(np.loadtxt(data_path + data_name + '_%d.txt'%(i+1))[:, :9])

    chain = np.concatenate(chain, axis=0)
    weights = chain[:,0]
    chain = chain[:, 2:]

    labels = [
            r"$\Omega_{\rm b}h^2$", 
            r"$\Omega_{\rm c}h^2$", 
            r"$100 \theta_{\rm MC}$",
            r"$\tau$", 
            r"$\Sigma m_\nu$",
            r"$n_{\rm s}$",
            r"$\ln(10^{10} A_{\rm s})$",
            ]

    fig = corner.corner(chain, bins=50, weights=weights,
            smooth=2., smooth1d=2,
            verbose=True,
            labels=labels, color='k', max_n_ticks=3)
            #truths=[m_true, b_true, np.log(f_true)])
    plt.show()

if __name__=="__main__":

    result_path = '/data/ycli/ksz/'
    result_name = 'pkSZ_result_tSZcln_DR13GPCM_50_AP7arcm_f1.41_sigz0.001tau_w_wa_chains.h5'

    analysis_mcmc(result_path, result_name)
    exit()

    data_path = '/data/ycli/planck_mcmc/base_mnu/planck_lowl_lowLike_highL/'
    data_name = 'base_mnu_planck_lowl_lowLike_highL'

    analysis_mcmc(result_path, result_name, 
            planck_path=data_path, planck_name=data_name, planck_n=8)

    #plot_planck_mcmc(data_path, data_name, n=8)





