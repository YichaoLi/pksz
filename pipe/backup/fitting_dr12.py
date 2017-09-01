from ksz.src import mcmc_fitting

ap = 7
result_path = '/data/ycli/ksz/'
result_name = 'pkSZ_result_tSZcln_DR12LOWZSwMASS_50_AP%darcm_f1.41.h5'%ap
output = result_path + result_name.replace('.h5', '_5p_chains.h5')
redshift = 0.2
logmass_min = 13.

mcmc_fitting.mcmc(result_path, result_name, nwalkers=200, steps=1000, output=output,
        redshift=redshift, logmass_min=logmass_min)

