#! /usr/bin/env python 
import numpy as np
from ksz.src import pairwise_ksz

#for ap_theta in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0]:
#for ap_theta in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
#for ap_theta in [14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]:

for ap_theta in [12.0, ]:

    ini_params = {
            'cat_sele'   : slice(None, None, None),
            'prefix'     : 'pkSZ_result_largemask',
            'rand_numb'  : 50,
            'AP_theta'   : ap_theta, # in arcmin
            'AP_f'       : 2.**0.5,
            'sigma_z'    : 0.01,
            'pl_map_code': 'tSZcln', #'HFI217'
            'pl_map_path': '/data/ycli/cmb/',
            #'pl_map_mask': 'masks_SEVEM.fits',
            'pl_map_mask': 'masks.fits',
            'pl_map_unit': 1.e-3, #K 
            'ga_cat_code': '6dFGPCM',
            'ga_cat_path': '/data/ycli/group_catalog/',
            'pksz_bins'  : np.linspace(10, 150, 21),
            'output'     : '/data/ycli/ksz/',
            'max_len'    : 1000,

            'cgc_criterion' : None,
            #'n_member_criterion_max' : 20,
            #'n_member_criterion_min' : 1,
            #'logHM_max' : 14.,
            'cmbrest'   : True,

            }
    pairwise_ksz.est_pksz_pipe(ini_params)

