import numpy as np
from ksz.src import plot
import matplotlib.pyplot as plt

result_path = '/data/ycli/ksz/'
output_path = '../png/'
#ap_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#ap_list = [6, 7, 8]
#ap_list = [8,]
#ap_list = [2, 3, 4, 5, 6, 7, 8]
#ap_list = [11, 10, 9, 8, 7, 6, 5]
#ap_list = [8, 7, 6, 5]
ap_list = [7, 8]

for ap in ap_list:
    result_name_list = [
            'pkSZ_result_newbin_tSZcln_DR13GPCM_JK_50_AP%darcm_f1.41_sigz0.001.h5'%ap,

            ]
    label_list = [
            'DR13 Group Catalogue, JK',
            ]
    text=r'$\theta_{\rm AP}=%d{\rm arcmin}\ \sigma_z=%4.3f$'%(ap, 0.001)

    plot.plot_jk(result_path, result_name_list, label_list=label_list,
            output_path=output_path, text=text, 
            ymax=0.12, ymin=-0.2) 


plt.show()
