#! /usr/bin/env python 

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
from mpi4py import MPI
import time

import read_data
import cosmology as cosmo
import utils
import units as u
import gc
import copy
import models

def est_pksz_pipe(ini_params, feedback=1):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0: feedback=0

    _s = ini_params['cat_sele'] 

    prefix = ini_params['prefix']

    N       = ini_params['rand_numb']
    theta   = ini_params['AP_theta']
    f       = ini_params['AP_f']
    sigma_z = ini_params['sigma_z']

    pksz_bins  = ini_params['pksz_bins']
    #pksz_bins -= 0.5 * (pksz_bins[1] - pksz_bins[0])

    output_path  = ini_params['output']
    
    pl_map = ini_params['pl_map_code']
    pl_data_path = ini_params['pl_map_path']
    pl = read_data.PLANCK_MAP(pl_data_path, feedback=feedback)
    pl.mask      = ini_params['pl_map_mask']
    pl.kSZ_map = 'planck_%s.fits'%pl_map
    pl.unit = ini_params['pl_map_unit']
    pl.unit = u.micro_K # convert to micro_K
    prefix += '_%s'%pl_map

    ga_cat = ini_params['ga_cat_code']
    ga_data_path = ini_params['ga_cat_path']
    ga = read_data.GAMA_CAT(ga_data_path, feedback=feedback)
    ga.catalog   = 'galcat_%s.dat'%ga_cat
    prefix += '_%s'%ga_cat

    ga.mask_bad(ga.catalog['Z']<0.01)

    if ini_params['cgc_criterion'] is not None:
        ga.mask_bad(ga.catalog['CGC'] < ini_params['cgc_criterion'])
        prefix += '_CGC%d'%ini_params['cgc_criterion']

    #if 'IO' in ga.catalog.dtype.names:
    #    ga.mask_bad(ga.catalog['IO']==0)
    #    prefix += '_I'

    #if ini_params['cmbrest'] == True:
    #    print "convert from cmbrest to heliocentric"
    #    prefix += '_HC'
    #    ga.catalog['Z'] = ga.cmbrest_to_heliocentric()

    if 'dec_criterion' in ini_params.keys():
        if ini_params['dec_criterion'] == 'N':
            ga.mask_bad(ga.catalog['DEC']<0)
            prefix += '_N'
        elif ini_params['dec_criterion'] == 'S':
            ga.mask_bad(ga.catalog['DEC']>0)
            prefix += '_S'

    #if 'n_member_criterion_max' in ini_params.keys():
    #    ga.mask_bad(ga.catalog['NMEM']>ini_params['n_member_criterion_max'])
    #    prefix += '_NMEM_Max%d'%ini_params['n_member_criterion_max']
    #if 'n_member_criterion_min' in ini_params.keys():
    #    ga.mask_bad(ga.catalog['NMEM']<ini_params['n_member_criterion_min'])
    #    prefix += '_NMEM_Min%d'%ini_params['n_member_criterion_min']

    if 'logHM_max' in ini_params.keys():
        if 'LOGHALOMASS' in ga.catalog.dtype.names:
            halo_mass = ga.catalog['LOGHALOMASS']
        elif 'LOGMASS' in ga.catalog.dtype.names:
            h = 0.67
            halo_mass = models.stellarM2haloM(10.**ga.catalog['LOGMASS']/h)*h
            halo_mass = np.log10(halo_mass)
        else:
            print "LOGHALOMASS or LOGMASS needed"
            exit()
        ga.mask_bad(halo_mass>ini_params['logHM_max'])
        #ga.mask_bad(ga.catalog['IO']==1)
        prefix += '_logHM_Max%d'%ini_params['logHM_max']

    if 'logHM_min' in ini_params.keys():
        if 'LOGHALOMASS' in ga.catalog.dtype.names:
            halo_mass = ga.catalog['LOGHALOMASS']
        elif 'LOGMASS' in ga.catalog.dtype.names:
            h = 0.67
            halo_mass = models.stellarM2haloM(10.**ga.catalog['LOGMASS']/h)*h
            halo_mass = np.log10(halo_mass)
        else:
            print "LOGHALOMASS or LOGMASS needed"
            exit()
        ga.mask_bad(halo_mass<ini_params['logHM_min'])
        #ga.mask_bad(ga.catalog['IO']==1)
        prefix += '_logHM_Min%d'%ini_params['logHM_min']

    jk_sample = False
    if 'jk' in ini_params.keys(): 
        if ini_params['jk']:
            prefix += '_JK'
            jk_sample = True

    sim_sample = False
    if 'sim' in ini_params.keys(): 
        if ini_params['sim']:
            prefix += '_SIM'
            sim_sample = True

    #if ga_cat == 'GAMA_CG':
    #    ga.mask_bad(ga.catalog['MAG_AUTO_R'] == 99.)
    #    ga.mask_bad(ga.catalog['MAG_AUTO_R'] >= 17.)
    #if ga_cat == 'DR12LOWZNCGC' and ini_params['zlim'] != []:
    #    zmin = ini_params['zlim'][0]
    #    zmax = ini_params['zlim'][1]
    #    zbad = np.logical_or(ga.catalog['Z'] < zmin, ga.catalog['Z'] >= zmax)
    #    ga.mask_bad(zbad)
    #    prefix += '_zlim%3.2fto%3.2f'%(zmin, zmax)
    #if ga_cat in ['DR12CMASNCGC', 'DR12CMASSCGC', 'DR12LOWZNCGC', 'DR12LOWZSCGC',
    #        'DR12CMASNCGC2', 'DR12CMASSCGC2', 'DR12LOWZNCGC2', 'DR12LOWZSCGC2']:
    #    mr = 22.5 - 2.5 * np.log10(ga.catalog['MODELFLUX_r'])
    #    ga.mask_bad(mr > ini_params['r_cut'])
    #    prefix += '_mlim%3.1f'%ini_params['r_cut']
    
    max_len = ini_params['max_len']

    ga_bad = np.ones(ga.catalog.shape[0])
    ga_bad[_s] = False
    ga.mask_bad(ga_bad)

    ga.rm_masked()

    c = cosmo.Cosmology()
    c = c.init_physical(ombh2=0.02230, omch2=0.1188, H0=67.74, omkh2=0.00037)
    ga.est_comoving_dist(c)
    ga.get_ga_coord()

    output_name = '%s_%d_AP%darcm_f%3.2f_sigz%4.3f'%(prefix, N, theta, f, sigma_z)
    pkSZ_estimator(pl, ga, theta=theta/60., f=f, sigma_z = sigma_z,
            feedback=feedback, max_len=max_len, 
            N=N, output_path=output_path, output_name=output_name, 
            pksz_bins=pksz_bins, rank=rank, size=size, comm=comm, 
            jk_sample=jk_sample, sim_sample=sim_sample)

def pkSZ_estimator(pl, ga, theta=5./60., f = 2.**0.5, sigma_z = 0.01,
        output_path=None, output_name='result', feedback=0, max_len=5000, N=10, 
        pksz_bins=None, comm=None, rank=0, size=1, 
        jk_sample=False, sim_sample=False):

    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()

    comm.barrier()
    if rank == 0:
        print "-"*80

    if output_path is not None:
        if rank == 0:
            result = h5py.File(output_path + output_name + '.h5', 'w')
            if feedback>0:
                print "Output path: %s.h5"%(output_path + output_name)
            result['ga_z']   = ga.catalog['Z']
            result['ga_dc']  = ga.catalog['DC']
            result['ga_ra']  = ga.catalog['RA']
            result['ga_dec'] = ga.catalog['DEC']
            result['ga_l']   = ga.catalog['L']
            result['ga_b']   = ga.catalog['B']
            #result['pl_map'] = pl.kSZ_map
    ga_mask = ga.mask

    if pksz_bins is None:
        pksz_bins = np.linspace(0, 150, 31)
    if rank == 0:
        result['pkSZ_bins'] = pksz_bins
    pksz_N = len(pksz_bins) - 1
    pksz = np.zeros((N+1, pksz_N))
    norm = np.zeros((N+1, pksz_N))
    numb = np.zeros((N+1, pksz_N))
    mask = np.zeros((N+1, ga_mask.shape[0]))

    ## rotating maps to generated mock catalog
    ## method used in previous analysis.
    #rots = np.random.random(N+1) * 360.
    #rots[0] = 0.
    #comm.Bcast(rots, root=0)
    #if rank == 0:
    #    result['rots'] = rots
    #comm.barrier()

    # using the same seed for all ranks
    #SEED = int(np.random.random() * (2**32 - 1))
    SEED = 3936650408
    #SEED = np.array(SEED, dtype='i')
    #comm.Bcast([SEED, MPI.INT], root=0)
    #comm.barrier()

    # to make sure each rank uses the same random map
    np.random.seed(SEED)

    if jk_sample:
        rd_num = 0
        jk_num = N
    else:
        rd_num = N
        jk_num = 0

    for i in range(rd_num+1): #[rank::size]:
        if feedback>0 and rank == 0:
            print "\nRANK:%02d: - %03d -"%(rank, i)
            time0 = time.time()
        ga.mask = copy.deepcopy(ga_mask)

        #if feedback>0 and rank == 0:
        #    print "RANK%02d: Estimating T_AP, rot %f degree"%(rank, rots[i])
        #t_ap = pl.ap_filter(ga.catalog['L'] + rots[i], ga.catalog['B'], 
        #        theta=theta, f=f, coord='G', degree=True, 
        #        rank = rank, size=size, comm=comm)
        if i != 0:
            if sim_sample:
                cls = np.load('/home/ycli/code/ksz/data/planck_cl.npy')
                ell = np.arange(cls.shape[0]).astype('float')
                #pl._map_random = sim.sim_cmb(cls, ell)
                pl.sim_map(cls, ell)
            else:
                pl.shuffle_map()

        t_ap = pl.ap_filter(ga.catalog['L'], ga.catalog['B'], theta=theta, 
                f=f, coord='G', degree=True, shuffle= i!=0, 
                rank = rank, size=size, comm=comm)
        t_ap_bad = t_ap == pl.UNSEEN
        ga.mask_bad(t_ap_bad)
        t_ap = t_ap[np.logical_not(t_ap_bad)]
        if feedback>0 and rank == 0:
            print "RANK%02d: Totally there are %d galaxies in the unmasked area"%(
                    rank, t_ap.shape[0])
        if rank == 0:
            result['t_ap_%03d'%i] = t_ap
            mask[i] = ga.mask

        if feedback>0 and rank == 0:
            print "RANK%02d: Estimating T_AP_bar"%rank
        t_ap_bar = pl.redshift_filter(t_ap, ga.catalog['Z'], sigma_z = sigma_z,
                max_len=max_len**2/len(t_ap) + 1, comm=comm, rank=rank, size=size)
        if rank == 0:
            result['t_ap_bar_%03d'%i] = t_ap_bar

        delta_t = t_ap - t_ap_bar

        if feedback>0 and rank == 0:
            print "RANK%02d: Estimating pkSZ"%rank

        dc      = ga.catalog['DC']
        ra      = ga.catalog['RA']  * np.pi / 180.
        dec     = ga.catalog['DEC'] * np.pi / 180.
        if jk_sample:
            pksz, norm, numb = get_pksz(pksz_bins, dc, ra, dec, delta_t, 
                    max_len=max_len, rank=rank, size=size, comm=comm, 
                    theta=theta*np.pi/180., jk_num=jk_num)
        else:
            pksz[i], norm[i], numb[i] = get_pksz(pksz_bins, dc, ra, dec, delta_t, 
                    max_len=max_len, rank=rank, size=size, comm=comm, 
                    theta=theta*np.pi/180.)
                    #theta=theta*np.pi/180.)
        comm.barrier()

        del t_ap, t_ap_bad, t_ap_bar, delta_t, dc, ra, dec
        gc.collect()

        if feedback>0 and rank == 0:
            time1 = time.time()
            print "TIMING: %f [s]"%(time1-time0)

    if comm != None:
        pksz_local = np.zeros((N+1, pksz_N))
        comm.Reduce(pksz, pksz_local, root=0)

        norm_local = np.zeros((N+1, pksz_N))
        comm.Reduce(norm, norm_local, root=0)

        numb_local = np.zeros((N+1, pksz_N))
        comm.Reduce(numb, numb_local, root=0)

        mask_local = np.zeros((N+1, ga_mask.shape[0]))
        comm.Reduce(mask, mask_local, root=0)

    if rank == 0:
        #print pksz_local
        norm_local[norm_local==0.] = np.inf
        pksz_local /= -norm_local
        norm_local[norm_local==np.inf] = 0.

        result['pkSZ']      = pksz_local[0,:]
        result['pkSZ_norm'] = norm_local[0,:]
        result['pkSZ_numb'] = numb_local[0,:]
        if jk_sample:
            result['pkSZ_jk']      = pksz_local[1:,:]
            result['pkSZ_norm_jk'] = norm_local[1:,:]
            result['pkSZ_numb_jk'] = numb_local[1:,:]
        else:
            result['pkSZ_random']      = pksz_local[1:,:]
            result['pkSZ_norm_random'] = norm_local[1:,:]
            result['pkSZ_numb_random'] = numb_local[1:,:]
        result['ga_mask'] = mask_local
        result.close()

    comm.barrier()



#@utils.log_timing
def get_pksz(pksz_bins, dc, ra, dec, delta_t, max_len=None, 
        rank=0, size=1, comm=None, theta=0., jk_num=0):

    pksz_N = len(pksz_bins) - 1
    pksz = np.zeros([jk_num + 1, pksz_N])
    norm = np.zeros([jk_num + 1, pksz_N])
    numb = np.zeros([jk_num + 1, pksz_N])
    hav = lambda x: (1. - np.cos(x)) / 2.

    cat_len = len(dc)
    if jk_num != 0: jk_len = cat_len / jk_num
    else: jk_len = 0
    loop_numb, loop_list = utils.gen_loop_N_better(cat_len, max_len)

    cos = np.cos
    sin = np.sin
    if rank == 0:
        print "RANK%02d: pksz looping numb %d"%(rank, loop_numb)
    for i in range(loop_numb)[rank::size]:

        index_st = loop_list[:i].sum()
        index_ed = loop_list[:i+1].sum()

        s_i = slice(None, index_ed , None)
        s_j = slice(index_st, index_ed, None)

        th_i = dec[s_i][:, None]
        ph_i = ra[s_i][:, None]
        dc_i = dc[s_i][:, None]

        th_j = dec[s_j][None, :]
        ph_j = ra[s_j][None, :]
        dc_j = dc[s_j][None, :]

        cos_theta = hav(th_i-th_j) + cos(th_i) * cos(th_j) * hav(ph_i - ph_j)
        cos_theta = 1. - 2. * cos_theta
        cos_theta[np.abs(cos_theta) < 1.e-9] = 0.

        cij = np.sqrt(dc_i*dc_i + dc_j*dc_j - 2.*dc_i*dc_j*cos_theta)
        cij[cij==0] = np.inf
        cij = (dc_i - dc_j) * (1. + cos_theta) / 2. / cij

        # kick out the overlaped galaxies
        kick_out = np.abs(cos_theta) > np.abs(cos(2*theta))
        #print "There are %d/%d overlap galaxies within %f arcmin"\
        #        %(np.sum(kick_out), np.prod(kick_out.shape), theta*180./np.pi * 60.)
        cij[kick_out] = 0.

        tij = delta_t[s_i][:, None] - delta_t[s_j][None, :]

        rij = np.sqrt((dc_i*cos(th_i)*cos(ph_i) - dc_j*cos(th_j)*cos(ph_j))**2 \
                + (dc_i*cos(th_i)*sin(ph_i) - dc_j*cos(th_j)*sin(ph_j))**2 \
                + (dc_i*sin(th_i) - dc_j*sin(th_j))**2)

        cij = np.triu(cij, k=1-loop_list[:i].sum())
        nij = np.ones(cij.shape).astype(float)
        nij[cij==0] = 0.

        for j in range(jk_num):

            cij_jk = copy.deepcopy(cij)
            tij_jk = copy.deepcopy(tij)
            rij_jk = copy.deepcopy(rij)
            nij_jk = copy.deepcopy(nij)

            cij_jk = utils.gen_jk_sample(j, jk_len, cij_jk, index_st)
            nij_jk[cij_jk==0] = 0.

            cij_jk = cij_jk.flatten()
            tij_jk = tij_jk.flatten()
            rij_jk = rij_jk.flatten()
            nij_jk = nij_jk.flatten()

            norm[j+1] += np.histogram(rij_jk, pksz_bins, weights=cij_jk**2)[0]
            pksz[j+1] += np.histogram(rij_jk, pksz_bins, weights=tij_jk*cij_jk)[0]
            numb[j+1] += np.histogram(rij_jk, pksz_bins, weights=nij_jk)[0]

            del cij_jk, rij_jk, tij_jk, nij_jk

        cij = cij.flatten()
        tij = tij.flatten()
        rij = rij.flatten()
        nij = nij.flatten()

        norm[0] += np.histogram(rij, pksz_bins, weights=cij**2)[0]
        pksz[0] += np.histogram(rij, pksz_bins, weights=tij*cij)[0]
        numb[0] += np.histogram(rij, pksz_bins, weights=nij)[0]

        del th_i, ph_i, dc_i, th_j, ph_j, dc_j, cos_theta, 
        del cij, rij, tij, nij, kick_out
        gc.collect()

    return pksz, norm, numb


if __name__ == '__main__':

    ini_params = {
            'cat_sele'   : slice(None, None, None),
            'prefix'     : 'pkSZ_result_ko',
            'rand_numb'  : 1,
            'AP_theta'   : 8., # in arcmin
            'AP_f'       : 2.**0.5,
            'pl_map_code': 'HSEVEM', # 'tSZcln'
            'pl_map_path': '/data/ycli/cmb/',
            'pl_map_mask': 'masks_SEVEM.fits',
            'pl_map_unit': 1., #K 1.e-3K
            'ga_cat_code': 'GAMA_CG', # 'cgc', '6df'
            'ga_cat_path': '/data/ycli/gama/',
            'pksz_bins'  : np.linspace(10, 150, 21),
            'output'     : '/data/ycli/ksz/',
            'max_len'    : 5200,
            }

    est_pksz_pipe(ini_params)

