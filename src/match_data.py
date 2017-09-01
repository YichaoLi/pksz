import numpy as np
from numpy.lib.recfunctions import append_fields
import read_data
import cosmology as cosmo

from mpi4py import MPI

def convert_mass_unit(catalog_path, catalog, h=0.7, mode=1):

    gc = read_data.GAMA_CAT(catalog_path)
    gc.catalog = catalog

    m_stellar = gc.catalog['LOGMASS']
    gc.add_fields('LOGMASS_RAW', m_stellar)

    if mode == 1:
        #m_stellar[gc.catalog['MC_MATCH'] == 1] -= np.log10(h)
        m_stellar -= np.log10(h)
    elif mode == 2:
        m_stellar[gc.catalog['MC_MATCH'] == 1] -= np.log10(h)
        m_stellar[gc.catalog['MC_MATCH'] == 2] += np.log10(0.7**2/h)
    gc._catalog['LOGMASS'] = m_stellar
    gc.save(overwrite=True)

def est_stellar_mass_6df(catalog_path, catalog):

    gc = read_data.GAMA_CAT(catalog_path)
    gc.catalog = catalog

    c = cosmo.Cosmology()
    c = c.init_physical(ombh2=0.02230, omch2=0.1188, H0=67.74, omkh2=0.00037)
    gc.est_comoving_dist(c)

    dc = gc.catalog['DC'] * 1.e6

    MJ = gc.catalog['J']  - 5.* (np.log10(dc * (1. + gc.catalog['Z'])) -1.)
    C = gc.catalog['bJ'] - gc.catalog['rF']
    y = gc.catalog['Z'] / (1. + gc.catalog['Z'])
    colors = gc.catalog['bJ'] - gc.catalog['rF']
    m_stellar = - 0.57 * colors + 0.48 + (3.70 - MJ) / 2.5

    m_stellar -= np.log10(c.H0 / 100.)

    gc.add_fields('LOGMASS', m_stellar)
    gc.add_fields('COLOR', colors)
    gc.save(overwrite=True)


def mass_to_logmass(catalog_path, catalog):

    gc = read_data.GAMA_CAT(catalog_path)
    gc.catalog = catalog

    logmass = np.log10(gc.catalog['MASS'])
    gc.add_fields('LOGMASS', logmass)
    gc.del_fields('MASS')
    gc.save(overwrite=True)

def match_stellarmass_GSWLC(catalog_path, catalog, stellar_catalog_path, stellar_catalog, 
        output_name):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    gc = read_data.GAMA_CAT(catalog_path)
    gc.catalog = catalog

    no_mass = gc.catalog['MC_MATCH'] == 0
    no_mass_catalog = gc.catalog[no_mass]
    
    N = no_mass_catalog.shape[0]
    print N

    mc = read_data.GAMA_CAT(stellar_catalog_path)
    mc.catalog = stellar_catalog

    stellarmass = np.zeros([N, 7])
    stellarmass_local = np.zeros([N, 7])

    for i in range(N)[rank::size]:

        plate   = mc.catalog['PLATE'] == no_mass_catalog['PLATE'][i]
        mjd     = mc.catalog['MJD'] == no_mass_catalog['MJD'][i]
        fiberid = mc.catalog['FIBERID'] == no_mass_catalog['FIBERID'][i]

        match_index = np.logical_and.reduce((plate, mjd, fiberid))
        mass_info = mc.catalog[match_index]
        if len(mass_info) == 1:
            stellarmass[i] = list(mass_info[0])[-7:]
            #print stellarmass[i]
            #print no_mass_catalog[i]
        elif len(mass_info) == 0: 
            print "No match"
        else:
            print "More than one match"

    comm.Reduce(stellarmass, stellarmass_local, root=0)

    if rank == 0:

        stellarmass = stellarmass_local

        ra_check  = np.abs(stellarmass[:, 0] - no_mass_catalog['RA'])/no_mass_catalog['RA']
        dec_check = np.abs(stellarmass[:, 1] - no_mass_catalog['DEC'])/no_mass_catalog['DEC']
        z_check   = np.abs(stellarmass[:, 2] - no_mass_catalog['Z'])/no_mass_catalog['Z']

        check = np.logical_or.reduce(
                [ra_check > 0.01, dec_check > 0.01, z_check > 0.01, 
                    stellarmass[:,3] == -99.])

        if np.any(check):

            print "BAD MATCH:"
            print stellarmass[check].shape


        gc._catalog['LOGMASS'][no_mass] = stellarmass[:,3] - np.log10(0.7)
        gc._catalog['MC_MATCH'][no_mass] = np.logical_not(check).astype('int') * 2.
        #gc.add_fields('LOGMASS_GSWLC',  stellarmass[:,3])
        #gc.add_fields('LOGMASSE_GSWLC', stellarmass[:,4])
        #gc.add_fields('MC_MATCH_GSWLC', )
        
        gc.save(output_name, overwrite=True)

        #names = gc.catalog.dtype.names
        #print names
        #header = '%20s '*len(names)
        #header = header%names
        #header = header[2:]
        #np.savetxt(output_name, gc.catalog, fmt='%20.12f', header=header)

    comm.barrier()

def match_stellarmass(catalog_path, catalog, stellar_catalog_path, stellar_catalog, 
        output_name):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    gc = read_data.GAMA_CAT(catalog_path)
    gc.catalog = catalog
    
    N = gc.catalog.shape[0]

    mc = read_data.GAMA_CAT(stellar_catalog_path)
    mc.catalog = stellar_catalog

    stellarmass = np.zeros([N, 6])
    stellarmass_local = np.zeros([N, 6])

    for i in range(N)[rank::size]:

        plate   = mc.catalog['PLATE'] == gc.catalog['PLATE'][i]
        mjd     = mc.catalog['MJD'] == gc.catalog['MJD'][i]
        fiberid = mc.catalog['FIBERID'] == gc.catalog['FIBERID'][i]

        match_index = np.logical_and.reduce((plate, mjd, fiberid))
        mass_info = mc.catalog[match_index]
        if len(mass_info) == 1:
            stellarmass[i] = list(mass_info[0])[-6:]
        elif len(mass_info) == 0: 
            print "No match"
        else:
            print "More than one match"

    comm.Reduce(stellarmass, stellarmass_local, root=0)

    if rank == 0:

        stellarmass = stellarmass_local

        ra_check  = np.abs(stellarmass[:, 0] - gc.catalog['RA'])/gc.catalog['RA']
        dec_check = np.abs(stellarmass[:, 1] - gc.catalog['DEC'])/gc.catalog['DEC']
        z_check   = np.abs(stellarmass[:, 2] - gc.catalog['Z'])/gc.catalog['Z']

        check = np.logical_or.reduce(
                [ra_check > 0.0001, dec_check > 0.0001, z_check > 0.0001])

        if np.any(check):

            print "BAD MATCH:"
            print stellarmass[check]

        gc.add_fields('LOGMASS',    stellarmass[:,3] - np.log10(0.7))
        gc.add_fields('MINLOGMASS', stellarmass[:,4] - np.log10(0.7))
        gc.add_fields('MAXLOGMASS', stellarmass[:,5] - np.log10(0.7))
        gc.add_fields('MC_MATCH', np.logical_not(check).astype('int'))
        
        gc.save(output_name)

        #names = gc.catalog.dtype.names
        #print names
        #header = '%20s '*len(names)
        #header = header%names
        #header = header[2:]
        #np.savetxt(output_name, gc.catalog, fmt='%20.12f', header=header)

    comm.barrier()


if __name__=="__main__":

    catalog_path = '/data/ycli/6df/'
    catalog = '6dFGS_2MASS_RA_DEC_Z_J_K_bJ_rF_GOOD.cat'
    #est_stellar_mass_6df(catalog_path, catalog)

    #exit()


    catalog_path = '/data/ycli/cgc/'
    catalog = 'CGC_wMASS.dat'
    #mass_to_logmass(catalog_path, catalog)
    convert_mass_unit(catalog_path, catalog, h=0.6774)

    exit()

    catalog_path = '/data/ycli/dr12/'
    catalog = 'galaxy_DR12v5_LOWZ_North_TOT.dat'
    stellar = 'portsmouth_stellarmass_passive_krou-DR12-boss_TOT.dat'
    output_name = catalog_path + 'galaxy_DR12v5_LOWZ_North_TOT_wMASS.dat'


    match_stellarmass(catalog_path, catalog, catalog_path, stellar, output_name)



