#! /usr/bin/env python 
import numpy as np
from ksz.src import pairwise_ksz


#for ap_theta in [2.0, 3.0, 4.0, 6.0, 7.0, 9.0, 12.0]:
for ap_theta in [5.0, 8.0, 10.0, 11.0]:

    ini_params = {
            'cat_sele'   : slice(None, None, None),
            'prefix'     : 'pkSZ_result',
            'rand_numb'  : 50,
            'AP_theta'   : ap_theta, # in arcmin
            'AP_f'       : 2.**0.5,
            'pl_map_code': 'tSZcln', #'HFI217'
            'pl_map_path': '/data/ycli/cmb/',
            'pl_map_mask': 'masks_SEVEM.fits',
            'pl_map_unit': 1.e-3, #K 
            'ga_cat_code': 'DR12LOWZNCGC', # 'CGC', '6dF'
            'ga_cat_path': '/data/ycli/dr12/',
            'pksz_bins'  : np.linspace(10, 150, 21),
            'output'     : '/data/ycli/ksz/',
            'max_len'    : 8000,
    
            #'r_cut'      : 17. + 2.5,
            'zlim'       : [0.3, 0.45]
    
            }
    
    pairwise_ksz.est_pksz_pipe(ini_params)
