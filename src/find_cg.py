import numpy as np

import cosmology as cosmo
import units as u
import read_data

from mpi4py import MPI

def find_GC(input_path, input_name, output_path=None, criterion=[1., 2., 4.]):

    prefix = ''

    if output_path is None:
        output_path = input_path

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ga = read_data.GAMA_CAT(input_path)
    ga.catalog = input_name

    c = cosmo.Cosmology()
    c = c.init_physical(ombh2=0.02230, omch2=0.1188, H0=67.74, omkh2=0.00037)
    ga.est_comoving_dist(c)

    cg_index = np.zeros(len(ga.catalog)) #.astype('bool')
    cg_index_local = np.zeros(len(ga.catalog)) #.astype('bool')
    
    cos = np.cos
    sin = np.sin
    
    h = c.H0/100.

    cat_z = ga.catalog['Z']
    criterion = np.sort(criterion)[::-1]
    
    for i in range(len(ga.catalog))[rank::size]:

        z_sel = []
    
        hav = lambda x: (1. - np.cos(x)) / 2.
    
        ra  = ga.catalog['RA'][i] * np.pi / 180.
        dec = ga.catalog['DEC'][i] * np.pi / 180.
        z   = ga.catalog['Z'][i]
    
        z_sel.append(np.abs(cat_z - z) * u.c < criterion[0] * 1000.*1.e3)
        z_sel[0][i] = False
        if not np.any(z_sel[0]):
            #print 'OK'
            cg_index[i] = criterion[0]
            continue

        for j in range(1, len(criterion)):
            z_sel.append(np.abs(cat_z[z_sel[0]] - z) * u.c < criterion[j] * 1000.*1.e3)
    
    
        ra_sel  = ga.catalog['RA'][z_sel[0]] * np.pi / 180.
        dec_sel = ga.catalog['DEC'][z_sel[0]] * np.pi / 180.
    
        cos_theta = hav(dec_sel - dec) + cos(dec_sel) * cos(dec) * hav(ra_sel-ra)
        cos_theta = 1. - 2. * cos_theta
        cos_theta[np.abs(cos_theta) < 1.e-9] = 0.
    
        cat_d = ga.catalog['DC'][z_sel[0]] / h * np.arccos(cos_theta)
        tr_sel = cat_d < 1. * criterion[0]
    
        if not np.any(tr_sel):
            #print 'OK'
            cg_index[i] = criterion[0]
            continue

        for j in range(1, len(criterion)):
            tr_sel = cat_d[z_sel[j]] < 1. * criterion[j]

            if not np.any(tr_sel):
                #print 'OK'
                cg_index[i] = criterion[j]
                #continue
                break

        #else:
        #    print 
    
    comm.Reduce(cg_index, cg_index_local, root=0)
    
    if rank == 0:
    
        #cg_index_local = cg_index_local.astype('bool')
        #catalog = ga.catalog[cg_index_local]
        for i in range(len(criterion)):
            print "Criterion %3.1f : %d"%( criterion[i], 
                    len(cg_index_local[cg_index_local >= criterion[i]]))
        ga.add_fields('CGC', cg_index_local)
        ga.save(overwrite=True)

    comm.barrier()
    

def find_GC_old(input_path, input_name, output_path=None, criterion=1., bcut=100):

    prefix = ''

    if output_path is None:
        output_path = input_path

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ga = read_data.GAMA_CAT(input_path)
    ga.catalog = input_name

    c = cosmo.Cosmology()
    c = c.init_physical(ombh2=0.02230, omch2=0.1188, H0=67.74, omkh2=0.00037)
    ga.est_comoving_dist(c)

    if bcut != 100:
        cat_n = len(ga.catalog)

        b = ga.catalog['MODELFLUX_r'] * 4. * np.pi * ga.catalog['DC']**2.
        b_bad = b < np.percentile(b, bcut)
        ga.mask_bad(b_bad)
        ga.rm_masked()
        print "%d of %d galaxies are removed due to the brightness cut"%(
                len(ga.catalog), cat_n)
        prefix += '_B%02d'%bcut
    
    cg_index = np.zeros(len(ga.catalog)) #.astype('bool')
    cg_index_local = np.zeros(len(ga.catalog)) #.astype('bool')
    
    cos = np.cos
    sin = np.sin
    
    h = c.H0/100.
    
    for i in range(len(ga.catalog))[rank::size]:
    
        hav = lambda x: (1. - np.cos(x)) / 2.
    
        ra  = ga.catalog['RA'][i] * np.pi / 180.
        dec = ga.catalog['DEC'][i] * np.pi / 180.
        z   = ga.catalog['Z'][i]
    
        z_sel = np.abs(ga.catalog['Z'] - z) * u.c < criterion * 1000.*1.e3
        z_sel[i] = False
    
        if not np.any(z_sel):
            #print 'OK'
            cg_index[i] = 1
            continue
    
        ra_sel  = ga.catalog['RA'][z_sel] * np.pi / 180.
        dec_sel = ga.catalog['DEC'][z_sel] * np.pi / 180.
    
        cos_theta = hav(dec_sel - dec) + cos(dec_sel) * cos(dec) * hav(ra_sel-ra)
        cos_theta = 1. - 2. * cos_theta
        cos_theta[np.abs(cos_theta) < 1.e-9] = 0.
    
        tr_sel = ga.catalog['DC'][z_sel] / h * np.arccos(cos_theta)
        tr_sel = tr_sel < 1. * criterion 
    
        if not np.any(tr_sel):
            #print 'OK'
            cg_index[i] = 1
        #else:
        #    print 
    
    comm.Reduce(cg_index, cg_index_local, root=0)
    
    if rank == 0:
    
        cg_index_local = cg_index_local.astype('bool')
        catalog = ga.catalog[cg_index_local]
    
    
        header = '%20s '*len(catalog.dtype.names)
        header = header%catalog.dtype.names
        header = header[2:]
        print header
    
        #np.savetxt('/data/ycli/gama/galcat_GAMA_CG.dat', catalog, fmt='%20.12f', 
        #        header=header)
        #np.savetxt('/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_RaDecZ_CGC.dat', 
        #        catalog, fmt='%20.12f', 
        #        header=header)
        #np.savetxt('/data/ycli/dr12/galaxy_DR12v5_LOWZ_South_RaDecZ_CGC.dat', 
        #        catalog, fmt='%20.12f', 
        #        header=header)
        #np.savetxt('/data/ycli/dr12/galaxy_DR12v5_CMASS_North_RaDecZ_CGC.dat', 
        #        catalog, fmt='%20.12f', 
        #        header=header)
        np.savetxt(output_path + input_name.split('.')[0] + '%s_CGC%d.dat'%(
            prefix, criterion), catalog, fmt='%20.12f', header=header)

def check_results(data_path, old_list):

    data = np.genfromtxt(data_path, names=True)
    
    #data_old = []
    #for old in old_list:
    #    data_tmp = np.genfromtxt(old, names=True)
    #    data_old.append(data_tmp)

    flag = data['CGC']
    cgc1 = flag >= 1
    cgc2 = flag >= 2
    cgc4 = flag >= 4

    print len(data[cgc1])#, len(data_old[0])
    print len(data[cgc2])#, len(data_old[1])
    print len(data[cgc4])#, len(data_old[2])

    for i in range(20):
        print data[cgc1][i]
        #print data_old[0][i]
        print

 
if __name__=="__main__":


    #input_path = '/data/ycli/dr12/'
    #input_name = 'galaxy_DR12v5_CMASS_North_RaDecZ.dat'
    #find_GC(input_path, input_name)


    data_path = '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_TOT_wMASS.dat'
    old_list = [
            '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_RaDecZ_CGC.dat',
            '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_RaDecZ_CGC2.dat',
            '/data/ycli/dr12/galaxy_DR12v5_LOWZ_North_RaDecZ_CGC4.dat',
            ]
    check_results(data_path, old_list)

