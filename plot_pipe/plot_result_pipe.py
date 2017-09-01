import numpy as np
from ksz.src import plot
from ksz.src import camb_wrap
import matplotlib.pyplot as plt

result_path = '/data/ycli/ksz/'
output_path = '../png/'
#ap_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#ap_list = [6, 7, 8]
#ap_list = [8,]
#ap_list = [2, 3, 4, 5, 6, 7, 8]
#ap_list = [11, 10, 9, 8, 7, 6, 5]
#ap_list = [8, 7, 6, 5]
ap_list = [5, 6, 7, 8, 9, 10, 11]

_camb = camb_wrap.CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')
_camb.set_matter_power(redshifts=[0.2, ], kmax=3.e1)

fitting_result = []

result_name_list = [
        #'pkSZ_result_newbin_tSZcln_DR12LOWZNwMASS_CGC1_50_AP%darcm_f1.41_sigz0.001.h5',
        #'pkSZ_result_newbin_tSZcln_DR12LOWZSwMASS_CGC1_50_AP%darcm_f1.41_sigz0.001.h5',
        #'pkSZ_result_newbin_tSZcln_DR13GPCM_50_AP%darcm_f1.41_sigz0.001.h5',

        #'pkSZ_result_newbin_tSZcln_DR12LOWZNwMASS_CGC1_JK_50_AP%darcm_f1.41_sigz0.001.h5',
        #'pkSZ_result_newbin_tSZcln_DR12LOWZSwMASS_CGC1_JK_50_AP%darcm_f1.41_sigz0.001.h5',
        #'pkSZ_result_newbin_tSZcln_DR13GPCM_JK_50_AP%darcm_f1.41_sigz0.001.h5'

        #'pkSZ_result_newbin_tSZcln_DR12CMASNwMASS_CGC1_JK_50_AP%darcm_f1.41_sigz0.001.h5',
        #'pkSZ_result_newbin_tSZcln_DR12CMASSwMASS_CGC1_JK_50_AP%darcm_f1.41_sigz0.001.h5',

        'pkSZ_result_newbin_tSZcln_DR12LOWZNwMASS_CGC1_SIM_50_AP%darcm_f1.41_sigz0.001.h5',
        ]
label_list = [

        #'DR12 LowZ North, CGC',
        #'DR12 LowZ South, CGC',
        #'DR13 Group Catalogue',

        'LowZ North CGC',
        #'LowZ South CGC',
        #'DR13 Group',

        #'CMASS North CGC',
        #'CMASS South CGC',
        ]
logmass_list = [
        [13., 16.], #[11.5, 13],
        #[13., 16.], #[11.5, 13],
        #[11., 14.],  #[11, 14],

        #[13., 16.], #[11.5, 13],
       # [13., 16.], #[11.5, 13],
        ]

#output_name = 'plot_pkSZ_CMASS_newbin_JK_AP'
#output_name = 'plot_pkSZ_result_newbin_JK_AP'
#output_name = 'plot_pkSZ_result_newbin_AP'
output_name = 'plot_pkSZ_result_newbin_SIM_AP'
plot.plot_pkSZ_ap(result_path, result_name_list, ap_list, label_list=label_list,
            output_path=output_path, output_name=output_name,
            camb=_camb, logmass_list = logmass_list,
            ymax=0.12, ymin=-0.19, xmax=150) 
plt.show()
exit()

for ap in ap_list:

    #output_name = 'plot_pkSZ_result_LowzN_Mass_AP%d'%ap
    output_name = 'plot_pkSZ_result_newbin_JK_AP%d'%ap
    text=r'$\theta_{\rm AP}=%d{\rm arcmin}\ \sigma_z=%4.3f$'%(ap, 0.001)

    result_name_list_ap = [x%ap for x in result_name_list]
    #plot.check_TAP_bar(result_path, result_name_list_ap,  label_list=label_list,
    #        output_path=output_path, output_name=output_name, text=text)
    #plot.plot_error(result_path, result_name_list_ap, label_list=label_list,
    #        output_path=output_path, output_name=output_name,text=text, 
    #        ymax=0.2, ymin=0.0)

    #result = plot.plot_pkSZ(result_path, result_name_list_ap, label_list=label_list,
    #        output_path=output_path, output_name=output_name,text=text,
    #        camb=_camb, logmass_list = logmass_list,
    #        #ymax=0.15, ymin=-0.3)
    #        ymax=0.12, ymin=-0.25) 
    #fitting_result.append(result)

    plot.plot_cov(result_path, result_name_list_ap, label_list=label_list,
            output_path=output_path, text=text)

#fitting_result = np.array(fitting_result)
#plot.plot_fitting_result_vs_ap(fitting_result, ap_list, label_list, 
#        output_path='../png/', output_name='plot_pkSZ_result_newbin')

plt.show()
