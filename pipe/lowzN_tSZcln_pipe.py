#! /usr/bin/env python 
import numpy as np
from ksz.src import pairwise_ksz

bins = np.concatenate([np.linspace(10, 80, 11), np.linspace(80, 150, 6)[1:]])

#for ap_theta in [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
for ap_theta in [7.0, 8.0]:

    ini_params = {
            'cat_sele'   : slice(None, None, None),
            'prefix'     : 'pkSZ_result_newbin',
            'rand_numb'  : 100,
            'AP_theta'   : ap_theta, # in arcmin
            'AP_f'       : 2.**0.5,
            'sigma_z'    : 0.001,
            'pl_map_code': 'tSZcln', #'HFI217'
            'pl_map_path': '/data/ycli/cmb/',
            #'pl_map_mask': 'masks_SEVEM.fits',
            'pl_map_mask': 'masks.fits',
            'pl_map_unit': 1.e-3, #K 
            'ga_cat_code': 'DR12LOWZNwMASS', 
            'ga_cat_path': '/data/ycli/dr12/',
            #'pksz_bins'  : np.linspace(10, 150, 21),
            'pksz_bins'  : bins,
            'output'     : '/data/ycli/ksz/',
            'max_len'    : 2000,

            'cgc_criterion' : 1.,
            'jk'         : True,
            'sim'        : False,
            }
    
    pairwise_ksz.est_pksz_pipe(ini_params)

    #ini_params = {
    #        'cat_sele'   : slice(None, None, None),
    #        'prefix'     : 'pkSZ_result_newbin',
    #        'rand_numb'  : 50,
    #        'AP_theta'   : ap_theta, # in arcmin
    #        'AP_f'       : 2.**0.5,
    #        'sigma_z'    : 0.001,
    #        'pl_map_code': 'tSZcln', #'HFI217'
    #        'pl_map_path': '/data/ycli/cmb/',
    #        #'pl_map_mask': 'masks_SEVEM.fits',
    #        'pl_map_mask': 'masks.fits',
    #        'pl_map_unit': 1.e-3, #K 
    #        'ga_cat_code': 'DR12LOWZNwMASS', 
    #        'ga_cat_path': '/data/ycli/dr12/',
    #        'pksz_bins'  : bins
    #        'output'     : '/data/ycli/ksz/',
    #        'max_len'    : 2000,

    #        'cgc_criterion' : 1.,
    #        'logHM_max' : 14.144,
    #
    #        }
    #
    #pairwise_ksz.est_pksz_pipe(ini_params)

    #ini_params = {
    #        'cat_sele'   : slice(None, None, None),
    #        'prefix'     : 'pkSZ_result_newbin',
    #        'rand_numb'  : 50,
    #        'AP_theta'   : ap_theta, # in arcmin
    #        'AP_f'       : 2.**0.5,
    #        'sigma_z'    : 0.001,
    #        'pl_map_code': 'tSZcln', #'HFI217'
    #        'pl_map_path': '/data/ycli/cmb/',
    #        #'pl_map_mask': 'masks_SEVEM.fits',
    #        'pl_map_mask': 'masks.fits',
    #        'pl_map_unit': 1.e-3, #K 
    #        'ga_cat_code': 'DR12LOWZNwMASS', 
    #        'ga_cat_path': '/data/ycli/dr12/',
    #        'pksz_bins'  : bins
    #        'output'     : '/data/ycli/ksz/',
    #        'max_len'    : 2000,
     
    #        'cgc_criterion' : 1.,
    #        'logHM_min' : 14.144,
    #
    #        }
    #
    #pairwise_ksz.est_pksz_pipe(ini_params)
