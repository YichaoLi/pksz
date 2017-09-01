import numpy as np
from ksz.src import mcmc_fitting

#covmat = np.loadtxt('/home/ycli/code/ksz/data/taubar_w0_wa.covmat')
covmat = None
ap = 8
result_path = '/data/ycli/ksz/'
result_name = 'pkSZ_result_newbin_tSZcln_DR13GPCM_JK_50_AP%darcm_f1.41_sigz0.001.h5'%ap
output = result_path + result_name.replace('.h5', 'tau_w_wa_chains.h5')
redshift = 0.1
logmass_min = 11.
logmass_max = 14.

param_dict = {
        'tau' : [ 1., -1., 3., r'\bar{\tau}'],
        'w'   : [-1., -3., 1., r'w'  ],
        'wa'  : [ 0., -3., 3., r'w_a'],
        }

mcmc_fitting.mcmc(result_path, result_name, param_dict,
        nwalkers=200, steps=1000, output=output,
        redshift=redshift, logmass_max=logmass_max, logmass_min=logmass_min, 
        covmat=covmat)

