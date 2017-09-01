import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import scipy as sp

import read_data

ga = read_data.GAMA_CAT('/data/ycli/dr12/')
ga.catalog = 'galcat_DR12LOWZNwMASS.dat'
ga.mask_bad(ga.catalog['Z']<0.01)
ga.rm_masked()
ga.get_ga_coord()

ra  = ga.catalog['L'] * np.pi / 180.
dec = np.pi/2. - ga.catalog['B'] * np.pi / 180.

pix = hp.ang2pix(nside=8, theta=dec, phi=ra)

pix_unique = sp.unique(pix)

print pix_unique.shape

hp.mollview(title='JK subsamples', fig=1, coord='C', rot=(180,0,0))
hp.graticule(dpar=30., dmer=30., coord='C')

for i in range(pix_unique.shape[0]):

    sel = pix == pix_unique[i]

    ra  = ga.catalog['RA'][sel]
    dec = ga.catalog['DEC'][sel]

    radec = np.concatenate([ra[None,:], dec[None,:]], axis=0)
    hp.projscatter(radec, lonlat=True, coord=['C'], 
            edgecolors=None, s=5, alpha=0.1)

plt.show()
