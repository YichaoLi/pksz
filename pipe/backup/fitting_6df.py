import numpy as np
from ksz.src import mcmc_fitting

# 6df
covmat = np.loadtxt('/home/ycli/code/ksz/data/taubar_mnu_omch2_w0_wa.covmat')
ap = 13
result_path = '/data/ycli/ksz/'
result_name = 'pkSZ_result_tSZcln_6dFGSDR3_50_AP%darcm_f1.41.h5'%ap
output = result_path + result_name.replace('.h5', '_4p_wcov_chains.h5')
redshift = 0.05
logmass_min = 11.

mcmc_fitting.mcmc(result_path, result_name, nwalkers=200, steps=1000, output=output,
        redshift=redshift, logmass_min=logmass_min, covmat=covmat)

#output_path = '/data/ycli/ksz/'
#output_name = 'vij_mnu'
#mnu_range = (0.0, 5.0, 200)
#ini_param_file = '/home/ycli/code/ksz/src/data/params.ini'
#
#analysis.est_velocitys(output_path=output_path, output_name=output_name, 
#        mnu_range=mnu_range, ini_param_file=ini_param_file)

#ini_params = {
#        'mnu': (0.0, 10.0, 100), #eV
#        }
#result_path = '/data/ycli/ksz/'
#result_name = 'pkSZ_result_tSZcln_DR12LOWZNCGC_50_AP5arcm.h5'
#analysis.pksz_analysis(result_path, result_name, ini_params, 
#        ini_param_file='/home/ycli/code/ksz/src/data/params.ini')
