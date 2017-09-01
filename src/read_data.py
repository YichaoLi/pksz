#! /usr/bin/env python 

import numpy as np
from numpy.lib.recfunctions import append_fields
from numpy.lib.recfunctions import drop_fields
import healpy as hp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cosmology as cosmo
#import aipy
import ephem as e
import units as u
import copy

import utils
import sim

class GAMA_CAT(object):
    '''
    read and manage the GAMA catalog
    '''

    def __init__(self, data_root=None, feedback=0):

        self._data_path = data_root
        self._catalog = None
        self._z_kind  = 'Z_TONRY'
        self.coord = 'eq'
        self.epoch = e.J2000
        self._mask = None
        self._feedback = feedback
    @property
    def mask(self):
        return self._mask

    @property
    def unmasked(self):
        return np.logical_not(self._mask)

    @mask.setter
    def mask(self, bad):
        self._mask = bad

    def mask_bad(self, sele):
        if len(sele) == len(self._mask):
            self._mask = np.logical_or(self._mask, sele)
        else:
            self._mask[self.unmasked] = sele
        #if self._catalog is not None:
        #    self._catalog = self._catalog[np.logical_not(sele)]

    def rm_masked(self):

        self._catalog = self._catalog[self.unmasked]
        self._mask    = self._mask[self.unmasked]

    @property
    def z_kind(self):
        return self._z_kind

    @z_kind.setter
    def z_kind(self, kind):
        self._z_kind = kind
        if self._catalog is not None:
            self._catalog['Z'] = self._catalog[self._z_kind]
            if self._feedback > 1:
                print "Note: set \'Z\' fields to \'%s\'."%self._z_kind

    @property
    def data_path(self):
        if self._data_path is None:
            print "Warnig: data_path not defined, return \'./\' as default!!"
            self._data_path = './'
        return self._data_path

    @data_path.setter
    def data_path(self, data_root):
        if not os.path.exists(data_root):
            raise ValueError('data_path does not exist!! please check\n\'%s\''%data_root)
        self._data_path = data_root

    @property
    def catalog(self):
        if self._catalog is None:
            print "Warnig: catalog not defined, return None as default!!"
        return self._catalog[self.unmasked]

    @catalog.setter
    def catalog(self, data_name):
        self._data_name = data_name
        self._catalog = np.genfromtxt(self.data_path + data_name, names=True)
        #self._catalog = np.recfromtxt(self.data_path + data_name, names=True)
        if 'Z' in self._catalog.dtype.fields.keys(): 
            if self._feedback>1:
                print "Note: Catalog has \'Z\' fields. " + \
                    "Use \'%s\' to get the correct z."%self._z_kind
        else:
            self._catalog = append_fields(self._catalog, 'Z', 
                    self._catalog[self._z_kind])
            if self._feedback:
                print "Note: set \'Z\' fields to \'%s\'."%self._z_kind
        self.check_redshift()
        self._mask = np.zeros(self._catalog.shape).astype('bool')

    def check_redshift(self):
        if self._catalog is not None:
            z_good = self._catalog['Z'] >= 0
            self._catalog = self._catalog[z_good]

    def est_comoving_dist(self, cosmology=None):

        if cosmology is None:
            cosmology = cosmo.Cosmology()

        if self._catalog is not None:
            dc = cosmology.comoving_distance(self._catalog['Z'])
            if 'DC' in self._catalog.dtype.fields.keys():
                self._catalog['DC'] = dc
                if self._feedback>1:
                    print "Note: \'DC\' field overwrited !"
            else:
                self._catalog = append_fields(self._catalog, 'DC', dc)

    def get_ga_coord(self, **kw):

        if 'L' in self._catalog.dtype.names:
            return

        l, b = utils.convert_eq2ga(self._catalog['RA'], self._catalog['DEC'], 
                i_epoch=self.epoch, **kw)
        self._catalog = append_fields(self._catalog, 'L', l)
        self._catalog = append_fields(self._catalog, 'B', b)

    def add_fields(self, field_label, field):

        if field_label in self._catalog.dtype.names:
            print "Note: field already exists, change the field values"
            self._catalog[field_label] = field
        else:
            self._catalog = append_fields(self._catalog, field_label, field)

    def del_fields(self, field_label):

        self._catalog = drop_fields(self._catalog, field_label)

    def save(self, output_name=None, overwrite=False):

        if output_name is None: 
            if overwrite:
                print "Warning: Overwrite the input catalog"
                output_name = self.data_path + self._data_name
                print output_name
            else:
                print "output name needed"
                exit()

        names = self._catalog.dtype.names
        header = '%16s '*len(names)
        header = header%names
        header = header[2:]
        np.savetxt(output_name, self._catalog, fmt='%16.8f', header=header)

    def cmbrest_to_heliocentric(self, l_helio=263.85, b_helio=48.25, v_helio=368.):

        '''
        Heliocenter is moving with v=368km/s towards (l, b)=(263.85, 48.25)
        Bennett et. al. 2003
        '''

        b = b_helio * np.pi / 180.
        l = l_helio * np.pi / 180.

        #n = 1024
        #map_tmp = np.arange(hp.nside2npix(n))
        #theta, phi = hp.pix2ang(n, map_tmp, nest=False)
        #theta = np.pi / 2. - theta

        #hav = lambda x: (1. - np.cos(x)) / 2.
        #cos_theta = hav(theta - b) + np.cos(theta) * np.cos(b) * hav(phi - l)
        #cos_theta = 1. - 2. * cos_theta
        #cos_theta[np.abs(cos_theta) < 1.e-9] = 0.

        #z_diff = -v_helio * 1.e3 * cos_theta / u.c

        self.get_ga_coord()

        theta = self.catalog['B'] * np.pi / 180.
        phi   = self.catalog['L'] * np.pi / 180.

        hav = lambda x: (1. - np.cos(x)) / 2.
        cos_theta = hav(theta - b) + np.cos(theta) * np.cos(b) * hav(phi - l)
        cos_theta = 1. - 2. * cos_theta
        cos_theta[np.abs(cos_theta) < 1.e-9] = 0.

        z_diff = -v_helio * 1.e3 * cos_theta / u.c

        #fig = plt.figure(10, figsize=(8, 5))
        #hp.mollview(title='Angle to heliocenter moving direction', fig=10, coord='G')
        #hp.graticule(dpar=30., dmer=30., coord='C')
        #lb = np.concatenate(
        #        [self._catalog['L'][None,:], self._catalog['B'][None,:]], axis=0)
        #hp.projscatter(lb, lonlat=True, edgecolors=None, c=z_diff, s=10,
        #        alpha=0.1)

        return self.catalog['Z'] + z_diff

        #plt.show()


class PLANCK_MAP(object):

    '''
    read and manage the Planck maps

    '''

    def __init__(self, data_root=None, feedback=0):

        self._data_path = data_root

        self._nest = False
        self._map_coord = 'G'
        self._map_ksz = None
        self._map_nos = None
        self._map_msk = None
        self._map_random = None
        self._unit = None
        self._feedback = feedback

        self.UNSEEN = hp.UNSEEN

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, temp_unit):
        if self._unit is None:
            self._unit = temp_unit
            if self._feedback > 1:
                print "Note: set map unit to %f K"%temp_unit
        else:
            if self._map_ksz is not None:
                if self._feedback > 1:
                    print "Note: convert map unit from %f to %f K"\
                            %(self._unit, temp_unit)
                self._map_ksz[self._map_ksz != hp.UNSEEN ] *= self._unit / temp_unit
            if self._map_nos is not None:
                if self._feedback > 1:
                    print "Note: convert noise unit from %f to %f K"\
                            %(self._unit, temp_unit)
                self._map_nos[self._map_ksz != hp.UNSEEN ] *= self._unit / temp_unit
            self._unit = temp_unit

    @property
    def data_path(self):
        if self._data_path is None:
            print "Warnig: data_path not defined, return \'./\' as default!!"
            self._data_path = './'
        return self._data_path

    @data_path.setter
    def data_path(self, data_root):
        if not os.path.exists(data_root):
            raise ValueError('data_path does not exist!! please check\n\'%s\''%data_root)
        self._data_path = data_root

    @property
    def mask(self):
        if self._map_msk is None:
            print "Warnig: Map mask not defined, return None as default!!"
        return self._map_msk

    @mask.setter
    def mask(self, data_name):
        self._map_msk = hp.read_map( self.data_path + data_name, 
                field=0, nest=self._nest, verbose=self._feedback)

    @property
    def kSZ_map(self):
        if self._map_ksz is None:
            print "Warnig: Map kSZ not defined, return None as default!!"
        return self._map_ksz

    @kSZ_map.setter
    def kSZ_map(self, data_name):
        self._map_ksz = hp.read_map(self.data_path + data_name, 
                field=0, nest=self._nest, verbose=self._feedback)
        self.apply_mask(self._map_ksz)
        #print np.min(np.abs(self._map_ksz))
        #self._map_ksz[np.abs(self._map_ksz) < 1.00408215076e-2] = hp.UNSEEN
        #self._map_ksz = hp.ma(self._map_ksz)
        self._nside = hp.get_nside(self._map_ksz)
        self._npix  = hp.get_map_size(self._map_ksz)
    
    @property
    def random_map(self):
        if self._map_random is None:
            print "Warnig: Map random not defined, return None as default!!"
        return self._map_random

    #@utils.log_timing
    def shuffle_map(self, seed=None):
        if self._map_random is None:
            self._map_random = copy.deepcopy(self._map_ksz)

        good = self._map_random[self._map_ksz != hp.UNSEEN]
        np.random.shuffle(good)
        self._map_random[self._map_ksz != hp.UNSEEN] = good
        pass

    def sim_map(self, cls, ell):
        self._map_random = sim.sim_cmb(cls, ell) / self._unit
        self._map_random[self._map_ksz == hp.UNSEEN] = hp.UNSEEN


    @property
    def kSZ_noise(self):
        if self._map_nos is None:
            print "Warnig: Map kSZ noise not defined, return None as default!!"
        return self._map_nos

    @kSZ_noise.setter
    def kSZ_noise(self, data_name):
        self._map_nos = hp.read_map(self.data_path + data_name, 
                field=0, nest=self._nest, verbose=self._feedback)
        self.apply_mask(self._map_nos)

    def apply_mask(self, pl_map):
        if self.mask is not None:
            pl_map[np.logical_not(self.mask.astype('bool'))] = hp.UNSEEN
            pl_map = hp.ma(pl_map)

    def check_map(self, catalog=None, ra=None, dec=None):

        if self.mask is not None:
            fig = plt.figure(1, figsize=(8, 5))
            hp.mollview(self.mask, title='MASK', fig=1, coord='G')
            hp.graticule(dpar=30., dmer=30., coord='C')
            fig.delaxes(fig.axes[1])

        if self.kSZ_map is not None:
            fig = plt.figure(2, figsize=(8, 5))
            hp.mollview(self.kSZ_map, title='kSZ', fig=2, coord='G')
            hp.graticule(dpar=30., dmer=30., coord='C')
        if catalog is not None:
            radec = np.concatenate([catalog['RA'][None,:], catalog['DEC'][None,:]], 
                    axis=0)
            hp.projscatter(radec, lonlat=True, coord=['C', 'G'], edgecolors=None, 
                    c='r', alpha=0.01, s=10)
        if ra is not None and dec is not None:
            radec = np.concatenate([ra[None,:], dec[None,:]], axis=0)
            hp.projscatter(radec, lonlat=True, coord='G', edgecolors=None, 
                    c='k', alpha=0.01, s=10)

        if self.kSZ_noise is not None:
            fig = plt.figure(3, figsize=(8, 5))
            hp.mollview(self.kSZ_noise, title='kSZ noise', fig=3, coord='G')
            hp.graticule(dpar=30., dmer=30., coord='E')
        
        if self.random_map is not None:
            fig = plt.figure(4, figsize=(8, 5))
            print self.random_map[1000:1020]
            hp.mollview(self.random_map, title='random', fig=4, coord='G')
            hp.graticule(dpar=30., dmer=30., coord='C')
            #fig.delaxes(fig.axes[1])

        #plt.show()

    def ap_filter(self, ra, dec, theta=5, f=1.414, coord='C', degree=True, shuffle=False,
            rank=0, size=1, comm=None):
        '''
        get the aperture photometry filtered temperature value

        ra, dec : the coordinate of the pixel, where the AP filter centered;
        theta   : aperture angular radius;
        f       : the outer radius of the surrounding ring is f*theta.

        '''

        cat_len = len(ra)

        kSZ_map = self.kSZ_map
        if shuffle:
            kSZ_map = self.random_map

        if degree:
            ra = ra * np.pi / 180.
            dec= dec* np.pi / 180.
            theta = theta * np.pi / 180.
        dec = np.pi/2. - dec
        if coord != self._map_coord:
            dec, ra = hp.Rotator(coord=(coord, self._map_coord))(dec, ra)

        ang_vec_list = hp.ang2vec(dec, ra)
        t_ap = np.zeros(cat_len)
        t_ap_local = np.zeros(cat_len)
        for i in range(cat_len)[rank::size]:
            pix_vec = hp.query_disc(self._nside, ang_vec_list[i], 
                    radius=theta, inclusive=False, fact=4, nest=self._nest)
            pix_vec_ring = hp.query_disc(self._nside, ang_vec_list[i], 
                    radius=theta*f, inclusive=False, fact=4, nest=self._nest)
            inner_sum   = np.sum(self.mask[pix_vec] * kSZ_map[pix_vec])
            inner_sum_n = np.sum(self.mask[pix_vec] )
            outer_sum   = np.sum(self.mask[pix_vec_ring] * kSZ_map[pix_vec_ring])
            outer_sum_n = np.sum(self.mask[pix_vec_ring] )
            #print "self.ap_filter %07d"%i
            #print "inner_sum=%17.8f, inner_sum_n=%d"%(inner_sum, inner_sum_n)
            #print "outer_sum=%17.8f, outer_sum_n=%d"%(outer_sum, outer_sum_n)
            if inner_sum_n == 0 or inner_sum_n == outer_sum_n:
                t_ap[i] = hp.UNSEEN
            else:
                t_ap[i] = inner_sum / inner_sum_n - \
                        (outer_sum - inner_sum)/(outer_sum_n - inner_sum_n)
        if comm is not None:
            comm.Allreduce(t_ap, t_ap_local)
            t_ap = t_ap_local
            comm.barrier()
        return t_ap

    #@utils.log_timing
    def redshift_filter(self, t_ap, z, sigma_z = 0.01, max_len=500, 
            rank=0, size=1, comm=None):

        loop = int(len(z) / max_len)
        sigma_z = 2.*sigma_z**2

        t_ap_bar = np.zeros(t_ap.shape)
        t_ap_bar_local = np.zeros(t_ap.shape)
        for i in range(loop+1)[rank::size]:
            s = slice(i*max_len, (i+1)*max_len, None)
            gauss_kern  = np.exp( -(z[:, None] - z[None, s])**2/sigma_z)
            t_ap_bar[s] = np.sum(t_ap[:, None] * gauss_kern, axis=0)
            t_ap_bar[s]/=np.sum(gauss_kern, axis=0)
            del gauss_kern
        if comm is not None:
            comm.Allreduce(t_ap_bar, t_ap_bar_local)
            t_ap_bar = t_ap_bar_local
            comm.barrier()

        return t_ap_bar

def read_dr12fits(data_name, output_name=None, fields=None):

    hdulist = pyfits.open(data_name)

    hdu = hdulist[1]

    dtype = []
    data  = []
    fmts  = []

    for i in range(hdu.header['TFIELDS']):

        label = hdu.header['TTYPE%d'%(i+1)]
        if fields is not None: 
            if not label in fields:
                continue
        col = hdu.data[label]
        if len(col.shape) == 1:
            col = col[:,None]
        print label, col.shape, hdu.header['TFORM%d'%(i+1)]
        if col.shape[1] == 1:
            #dtype.append((label, col.dtype.str))
            dtype.append('%16s'%label)
            fmts.append(fmt[hdu.header['TFORM%d'%(i+1)][-1]])
            data.append(col[:,0])
        elif col.shape[1] == 5:
            for j in range(5):
                #dtype.append(('%s_%s'%(label, j), col.dtype.str))
                dtype.append('%14s_%s'%(label, 'ugriz'[j]))
                fmts.append(fmt[hdu.header['TFORM%d'%(i+1)][-1]])
                data.append(col[:,j])
        else:
            for j in range(col.shape[1]):
                #dtype.append(('%s_%s'%(label, j), col.dtype.str))
                dtype.append('%14s_%d'%(label, j))
                fmts.append(fmt[hdu.header['TFORM%d'%(i+1)][-1]])
                data.append(col[:,j])

    len(data)
    data = np.rec.fromarrays(data, names=dtype)

    print data.shape
    print data.dtype
    print len(fmts)
    header = '%s '*len(dtype)
    header = header%tuple(dtype)
    header = header[2:]

    if output_name is not None:
        np.savetxt(output_name, data, fmt=fmts, header=header)
    return data, fmts, header

def plot_cats(cats_path_list, cats_name_list, output_path):

    color = 'mcrbgk'

    #fig = plt.figure(1, figsize=(8, 5))
    #hp.mollview(title='Galaxy Groups', fig=1, coord='G')
    #hp.graticule(dpar=30., dmer=30., coord='C')
    for i in range(len(cats_path_list)):
        cats_path = cats_path_list[i]
        cats_name = cats_name_list[i]
        fig = plt.figure(i+1, figsize=(8, 5))
        hp.mollview(title=cats_name.replace('_', ' '), fig=i+1, coord='G')
        hp.graticule(dpar=30., dmer=30., coord='C')
        ga = GAMA_CAT(cats_path)
        ga.catalog = cats_name
        s = 5
        if cats_name == 'galcat_6dFGPCM.dat':
            #ga.mask_bad(ga.catalog['NMEM'] )
            ga.mask_bad(ga.catalog['Z']<0.01)
            #ga.mask_bad(ga.catalog['NMEM']>1)
            #ga.mask_bad(ga.catalog['LOGHALOMASS']>14)
            s = ga.catalog['NMEM'] * 5
            ga.rm_masked()
        radec = np.concatenate(
                [ga.catalog['RA'][None,:], ga.catalog['DEC'][None,:]], axis=0)
        hp.projscatter(radec, lonlat=True, coord=['C', 'G'], edgecolors=None, 
                    c=color[i], s=s, alpha=0.1)

        z1 = ga.cmbrest_to_heliocentric(l_helio=263.85, b_helio=48.25, v_helio=368.)
        z2 = ga.cmbrest_to_heliocentric(l_helio=263.85, b_helio=48.25, v_helio=368.)

        print (z1-z2).max()
        print (z1-z2).min()

        plt.savefig(output_path + cats_name.replace('.dat', '.png'))


    plt.show()




if __name__=="__main__":

    test_planck = test_gama = test_cgc = test_g3c = test_cat = False

    #test_cat = True
    test_planck = True
    #test_gama   = True
    #test_cgc = True
    #test_g3c = True
    import time

    if test_cat:

        cats_path_list = [
                #'/data/ycli/group_catalog/',
                '/data/ycli/group_catalog/',
                #'/data/ycli/dr12/',
                #'/data/ycli/dr12/',
                #'/data/ycli/6df/',
                ]
        cats_name_list = [
                #'galcat_DR13GPCM.dat',
                'galcat_6dFGPCM.dat',
                #'galcat_DR12LOWZNwMASS.dat',
                #'galcat_DR12LOWZSwMASS.dat',
                #'galcat_6dFGSDR3.dat',
                ]
        output_path = '/home/ycli/workspace/ksz/png/'

        plot_cats(cats_path_list, cats_name_list, output_path)

    if test_planck:
        data_path = '/data/ycli/cmb/'
        pl = PLANCK_MAP(data_path)
        pl.mask      = 'masks.fits'
        #pl.mask      = 'COM_Mask_CMB-confidence-Tmask-IQU-sevem-field-Int_2048_R2.01_full.fits'
        #pl.mask      = 'COM_CMB_IQU-common-field-MaskInt_2048_R2.01.fits'
        pl.kSZ_map   = 'DX11d_2DILC_MAP_HFIonly_NOPS.fits'
        #pl.kSZ_map   = 'HFI_CompMap_Foregrounds-sevem_2048_R2.00.fits'
        pl.kSZ_noise = 'DX11d_2DILC_NOISE_HFIonly_NOPS.fits'
        pl.unit = u.milli_K
        pl.unit = u.micro_K
        #pl.shuffle_map()
        cls = np.load('/home/ycli/code/ksz/data/planck_cl.npy')
        ell = np.arange(cls.shape[0]).astype('float')
        np.random.seed(3936650408)
        pl.sim_map(cls, ell)
        pl.check_map()
        plt.show()
        exit()
        pl.sim_map(cls, ell)
        pl.check_map()
        exit()
        ra_list = np.array([-90., -180.])
        dec_list = np.array([0, -30])

        for i in range(len(ra_list)):
            print t_ap[i], t_ap_self[i]

    if test_cgc:

        data_path = '/data/ycli/cgc/'
        ga = GAMA_CAT(data_path)
        ga.catalog   = 'CGC_ra_dec_redshift.dat'

        ga.est_comoving_dist()

        print ga.catalog['RA'].shape
        print ga.catalog['Z'].shape
        print ga.catalog['DC'].shape

        ga.mask_bad(ga.catalog['Z']>0.1)

        print ga.catalog['RA'].shape
        print ga.catalog['Z'].shape
        print ga.catalog['DC'].shape

    if test_gama:
        data_path = '/data/ycli/gama/'
        ga = GAMA_CAT(data_path)
        #ga.catalog   = 'GAMA_TilingCat.dat'
        #print ga.catalog.dtype
        #ga.catalog   = 'GAMA_SpecObj_v08.dat'
        ga.catalog   = 'GAMA_DistancesFrames_ApMatchedCat.dat'
        print ga.catalog.dtype

        ga.mask_bad(ga.catalog['MAG_AUTO_R'] == 99.)
        #ga.mask_bad(ga.catalog['MAG_AUTO_R'] > 17.)
        ga.mask_bad(ga.catalog['MAG_AUTO_R'] < 17.)
        ga.mask_bad(ga.catalog['MAG_AUTO_R'] > 18.)
        print ga.catalog['MAG_AUTO_R'].min()
        print ga.catalog['MAG_AUTO_R'].max()
        print ga.catalog.shape

        exit()

        good = ga.catalog[:, 4]!=-9.99999
        catalog_1 = ga.catalog[good, :]

        good = ga.catalog[:, 6]==1
        catalog_2 = ga.catalog[good, :]

        print ga.catalog.shape
        print catalog_1.shape
        print catalog_2.shape

        print np.sum(np.abs(catalog_1[:,0] - catalog_2[:,0]))

        for i in range(10):
            print catalog_1[i, 0], catalog_2[i, 0]

    if test_g3c:

        data_path = '/data/ycli/gama/'
        ga = GAMA_CAT(data_path)
        ga.catalog   = 'galcat_G3C.dat'

