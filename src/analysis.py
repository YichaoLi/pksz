import numpy as np
import models
from models import CAMB

import h5py
from mpi4py import MPI

ini_params = {
        'mnu': (0.0, 10.0, 100), #eV
        }

def est_velocitys(output_path='./', output_name = 'vij', mnu_range=(0.0, 10.0, 100),
        ini_param_file='./data/params.ini'):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    camb = CAMB(ini_param_file=ini_param_file)
    zs = np.array([0.5, 0.2, 0.1])
    rh = np.linspace(10, 200, 200)
    mnu_list = np.linspace(*mnu_range)
    vij = np.zeros(mnu_list.shape + zs.shape + rh.shape)
    for i in range(vij.shape[0])[rank::size]:
        mnu = mnu_list[i]
        camb.params['omnuh2'] = mnu / 94.07
        camb.set_cosmology_params()
        camb.get_results()
        result_vij, rh = models.mean_pairwise_v(camb, rh=rh, redshifts=zs)
        vij[i,...] = result_vij

    vij_local = np.zeros_like(vij)
    comm.Reduce(vij, vij_local, root=0)
    vij = vij_local

    comm.barrier()
    if rank == 0:

        for i in range(zs.shape[0]):
            vij_results = h5py.File(output_path + output_name + "_z%3.2f.h5"%zs[i], 'w')
            vij_results['vij'] = vij[:,i,:]
            vij_results['rh']  = rh
            vij_results['nu']  = mnu_list
            vij_results.close()

    comm.barrier()

def pksz_analysis(result_path, result_name, ini_params, label_list=[], 
        output_path=None, output_name='', bins_shift=0.5, text='', 
        ymax=None, ymin=None, ini_param_file='./data/params.ini'):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        result = h5py.File(result_path + result_name, 'r')

        pksz = result['pkSZ'][:]
        pksz_random = result['pkSZ_random'][:]
        pksz_bins = result['pkSZ_bins'][:]
        d_bins = pksz_bins[1] - pksz_bins[0]
        pksz_bins = pksz_bins[:-1] + 0.5 * d_bins

        pksz_errs = np.std(pksz_random, axis=0)
    else:
        pksz_bins = None

    comm.barrier()
    pksz_bins = comm.bcast(pksz_bins, root=0)

    camb = CAMB(ini_param_file=ini_param_file)
    zs = np.array([0.2, ])

    mnu_list = np.linspace(*ini_params['mnu'])
    vij = np.zeros(mnu_list.shape + pksz_bins.shape)
    for i in range(vij.shape[0])[rank::size]:
        mnu = mnu_list[i] / 94.07
        camb.params['omnuh2'] = mnu
        camb.set_cosmology_params()
        camb.get_results()
        result_vij, rh = models.mean_pairwise_v(camb, rh=pksz_bins, redshifts=zs)
        vij[i] = result_vij[0]

    vij_local = np.zeros_like(vij)
    comm.Reduce(vij, vij_local, root=0)
    vij = vij_local

    comm.barrier()

    if rank == 0:
        tau_bar = np.linspace(0.0, 1.0, 100) * 1.e-4
        amp = tau_bar * 2.7255e6 / 2.99e5

        chisq  = (vij[:, None, :] * amp[None, :, None] - pksz[None, None, :])**2.
        chisq /= pksz_errs[None, None, :]**2.
        chisq  = np.sum(chisq, axis=-1)
        chisq -= np.min(chisq)

        #np.save(result_path + result_name.replace('.h5', 'chisq.h5'), chisq)
        chisq_result = h5py.File(
                result_path + result_name.replace('.h5', '_chisq.h5'), 'w')
        chisq_result['chisq'] = chisq
        chisq_result['x'] = tau_bar
        chisq_result['y'] = np.linspace(*ini_params['mnu'])
        chisq_result.close()

    comm.barrier()


if __name__=='__main__':

    result_path = '/data/ycli/ksz/'
    result_name = 'pkSZ_result_tSZcln_DR12LOWZNCGC_50_AP5arcm.h5'
    pksz_analysis(result_path, result_name, ini_params)
    exit()

