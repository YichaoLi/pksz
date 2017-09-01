import camb_wrap
import numpy as np
import healpy as hp
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

#_camb = camb_wrap.CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')

def sim_cmb(cls, ls, lmax=4000, fwhm=5.):

    fwhm *= np.pi / 180. / 60.

    sel = ls <= lmax

    cls = cls[sel] * 2.725**2.
    ls  = ls[sel]

    cls[1:] = cls[1:] / (ls[1:] * (ls[1:] + 1)) * 2. * np.pi

    ln = ls.shape[0]

    triu = np.triu_indices(ln)

    sigma = np.sqrt(cls[None, :]/2.)

    alm_r = np.random.randn(ln, ln) * sigma
    alm_r[0, :] *= 2.**0.5
    alm_r = alm_r[triu]
    
    alm_i = np.random.randn(ln, ln) * sigma
    alm_i[0, :] = 0.
    alm_i = alm_i[triu]

    alm = alm_r + alm_i * 1j

    cmb_map = hp.alm2map(alm, nside=2048, lmax=lmax, fwhm=fwhm)

    return cmb_map

#cls, ls = _camb.cmb_angular_power_spectrum()
#print ls.max()
#
#cmb_map = sim_cmb(cls, ls)
#
#std = 3. * np.std(cmb_map)
#
#hp.mollview(cmb_map, coord='G', min=-std, max=std)
#
#plt.show()
#   
