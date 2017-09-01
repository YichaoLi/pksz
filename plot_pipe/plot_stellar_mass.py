from ksz.src import plot
import matplotlib.pyplot as plt

data_path_list = [
        '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_TOT_wMASS.dat',
        '/data/ycli/dr12/galaxy_DR12v5_LOWZ_South_TOT_wMASS.dat',
        '/data/ycli/dr12/galaxy_DR12v5_CMASS_North_TOT_wMASS.dat',
        '/data/ycli/dr12/galaxy_DR12v5_CMASS_South_TOT_wMASS.dat',
        #'/data/ycli/6df/6dFGS_2MASS_RA_DEC_Z_J_K_bJ_rF_GOOD.cat',
        #'/data/ycli/group_catalog/6dFGS_M_group.dat',
        #'/data/ycli/group_catalog/6dFGS_L_group.dat',
        '/data/ycli/group_catalog/SDSS_M_group.dat',
        #'/data/ycli/group_catalog/SDSS_L_group.dat',
        '/data/ycli/cgc/CGC_wMASS.dat',
        ]
label_list = [
        'LOWZ North CGC',
        'LOWZ South CGC',
        'CMASS North',
        'CMASS South',
        #'6dF',
        #'6dF mass-weighted halo center',
        #'6dF luminosity-weighted halo center',
        'DR13 Group',
        #'dr13 luminosity-weighted halo center',
        'DR7 CGC',
        ]
ap_list = [
        7.,
        7.,
        #0.,
        #0.,
        8.,
        #11.,
        11.,
        #11.,
        7.,
        #7.,
        ]
#plot.plot_stellarmass_hist(data_path_list, label_list)

plot.plot_halomass_hist(data_path_list, label_list)
#plot.plot_rvir_hist(data_path_list, label_list, rho_crit = 2.775e11, ap_list=ap_list)

#plot.plot_z_hist(data_path_list, label_list)

plt.show()
