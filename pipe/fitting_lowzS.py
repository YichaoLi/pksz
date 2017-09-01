import numpy as np
from ksz.src import mcmc_fitting

#covmat = np.loadtxt('/home/ycli/code/ksz/data/taubar_w0_wa.covmat')
covmat = None
ap = 7
result_path = '/data/ycli/ksz/'
#result_name = 'pkSZ_result_tSZcln_DR13GPCM_50_AP%darcm_f1.41_sigz0.001.h5'%ap
result_name = 'pkSZ_result_tSZcln_DR12LOWZSwMASS_CGC1_50_AP%darcm_f1.41_sigz0.001.h5'%ap
output = result_path + result_name.replace('.h5', 'tau_chains.h5')
redshift = 0.1
logmass_min = 13.
logmass_max = 15.

mcmc_fitting.mcmc(result_path, result_name, nwalkers=200, steps=1000, output=output,
        redshift=redshift, logmass_max=logmass_max, logmass_min=logmass_min, 
        covmat=covmat)

