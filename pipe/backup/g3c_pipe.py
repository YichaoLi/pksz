#! /usr/bin/env python 
import numpy as np
from ksz.src import pairwise_ksz

ini_params = {
        'cat_sele'   : slice(None, None, None),
        'prefix'     : 'pkSZ_result',
        'rand_numb'  : 5,
        'AP_theta'   : 8., # in arcmin
        'AP_f'       : 2.**0.5,
        'pl_map_code': 'HFI217', # 'tSZcln'
        'pl_map_path': '/data/ycli/cmb/',
        'pl_map_mask': 'masks.fits',
        'pl_map_unit': 1., #K 1.e-3K
        'ga_cat_code': 'G3C', # 'CGC', '6dF'
        'ga_cat_path': '/data/ycli/gama/',
        'pksz_bins'  : np.linspace(10, 150, 21),
        'output'     : '/data/ycli/ksz/',
        'max_len'    : 5200,
        }

pairwise_ksz.est_pksz_pipe(ini_params)

ini_params = {
        'cat_sele'   : slice(None, None, None),
        'prefix'     : 'pkSZ_result',
        'rand_numb'  : 5,
        'AP_theta'   : 8., # in arcmin
        'AP_f'       : 2.**0.5,
        'pl_map_code': 'tSZcln', #'HFI217'
        'pl_map_path': '/data/ycli/cmb/',
        'pl_map_mask': 'masks.fits',
        'pl_map_unit': 1.e-3, #K 
        'ga_cat_code': 'G3C', # 'CGC', '6dF'
        'ga_cat_path': '/data/ycli/gama/',
        'pksz_bins'  : np.linspace(10, 150, 21),
        'output'     : '/data/ycli/ksz/',
        'max_len'    : 5200,
        }

pairwise_ksz.est_pksz_pipe(ini_params)
