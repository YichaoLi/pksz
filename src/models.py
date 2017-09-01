from scipy.special import p_roots
from scipy.integrate import quad
from scipy.integrate import fixed_quad
from scipy import interpolate
import numpy as np
import camb_wrap
import time
import utils
import plot
import matplotlib.pyplot as plt
import h5py

def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n] 
    _cached_roots_legendre.cache[n] = p_roots(n)
    return _cached_roots_legendre.cache[n]
_cached_roots_legendre.cache = dict()

_cached_roots_legendre(10000)

class HALO_MASS_FUNCTION(object):

    def __init__(self, pk=None, kh=None, redshifts=None, da=None, fa=None, 
            omm=0.2747, rho_crit = 2.775e11, delta_c = 1.686, npoints=10000):

        '''
        rho_crit = 2.775e11 h^2 M_sun Mpc^{-3}
        '''
        self.x, self.w = _cached_roots_legendre(npoints) #p_roots(npoints)

        self.pk = pk
        self.kh = kh
        self.z  = redshifts
        self.delta_c = delta_c
        self.da = da
        self.fa = fa

        if self.pk is not None:
            self.get_pk_at_fixed_kh(logspace=True)

        if self.z is not None:
            self.z = self.z[:, None]

        self.rhom_mean = omm * rho_crit * (1. + self.z)**3

        self.windowf = lambda x: 3.*(np.sin(x) - x*np.cos(x)) / x**3.
        self.r_func = lambda m: ( 3. * m / (4. * np.pi * self.rhom_mean))**(1./3.)

    def get_fixed_kh(self, kmin=None, kmax=None, logspace=True):

        #x, khw = p_roots(npoints)
        x = self.x
        khw = self.w

        if kmin is None:
            kmin = self.kh.min()
        if kmax is None:
            kmax = self.kh.max()
        if logspace:
            kmax = np.log10(kmax)
            kmin = np.log10(kmin)
        self.lgkh_wet = khw * (kmax - kmin) / 2.0
        self.lgkh_int = (kmax - kmin) * (x + 1.) / 2.0 + kmin

    def get_pk_at_fixed_kh(self, logspace=True):
        """
        in order to speed up the integration, fix kh at some points
        if logspace, dk = k ln10 d(lgk)
        """

        self.get_fixed_kh(self, logspace=True)

        pk_interp = interpolate.interp1d(np.log10(self.kh), self.pk, axis=1)
        self.pk_int = pk_interp(self.lgkh_int)

    #@utils.log_timing
    def sigma_square(self, mass):

        self.r_mass = self.r_func(mass[None, :])

        winf = self.windowf((10.**self.lgkh_int[None, None, :])*self.r_mass[:, :, None])

        result = np.sum( self.lgkh_wet[None, None, :]\
                * 10.**(3.*self.lgkh_int[None, None, :])\
                * np.log(10.) * self.pk_int[:, None, :] * winf**2, 
                axis=2) / 2. / np.pi**2

        return result

    #@utils.log_timing
    def sigma_square_old(self, mass):

        self.r_mass = self.r_func(mass[None, :])

        result = np.zeros(self.r_mass.shape)
        pkf = lambda lnk, i: np.interp(lnk, np.log(self.kh), self.pk[i])
        itf = lambda lnk, r, i: pkf(lnk, i) * self.windowf(np.exp(lnk)*r)**2.\
                * np.exp(lnk)**3. / 2. / np.pi**2
        kmin = np.log(self.kh.min())
        kmax = np.log(self.kh.max())
        sigma_square_int = np.vectorize(lambda r, i: \
                fixed_quad(itf, kmin, kmax, args=(r, i), n=10000)[0])
                #quad(itf, kmin, kmax, args=(r, i), limit=2000)[0])
        for i in range(len(self.z)):
            result[i] = sigma_square_int(self.r_mass[i], i)

        return result

    def d_lnsigmainv_dmass(self, mass, sigma_sq=None, dmass=None):

        if dmass is None:
            dmass = mass.min() * 1.e-2

        if sigma_sq is None:
            sigma_sq = self.sigma_square(mass)
        sigma_sq_diff = self.sigma_square(mass+dmass)

        #return np.log(sigma_sq/sigma_sq_diff) / dmass
        return - 1./np.sqrt(sigma_sq) * \
                (np.sqrt(sigma_sq_diff) - np.sqrt(sigma_sq)) / dmass

    def f_sigma(self, sigma_sq):

        return 0


    def dn_dmass(self, mass, dmass=None):
        #print "Estimate Mass Function"

        sigma_sq = self.sigma_square(mass)
        self.sigma_sq = sigma_sq

        return self.f_sigma(sigma_sq) * self.rhom_mean  / mass[None, :]\
                * self.d_lnsigmainv_dmass(mass, sigma_sq=sigma_sq, dmass=dmass)

    #@utils.log_timing
    def b_halo_slow(self, dndm=None, sigma_sq=None, mass=None, 
            logmass_min=10, logmass_max=16):

        #print "Estimate Halo Bias"

        #if dndm is None:
        #    mass = np.logspace(logmass_min, logmass_max, 100)
        #    dndm = self.dn_dmass(mass)
        #    sigma_sq = self.sigma_sq
        ##dndm_f = lambda lgm, i: np.interp(lgm, np.log10(mass), dndm[i,:])
        #dndm_f = interpolate.interp1d(np.log10(mass), dndm, axis=1)
        #sigma_sq_f = interpolate.interp1d(np.log10(mass), sigma_sq, axis=1)

        #x, w = p_roots(npoints)
        x = self.x
        w = self.w
        mmax = logmass_max
        mmin = logmass_min
        lgm_wet = w * (mmax - mmin) / 2.0
        lgm_int = (mmax - mmin) * (x + 1.) / 2.0 + mmin
        dndm_int = self.dn_dmass(10.**lgm_int)
        sigma_sq_int = self.sigma_sq
        #dndm_int = dndm_f(lgm_int)
        #sigma_sq_int = sigma_sq_f(lgm_int)

        r_mass = self.r_func(10.**lgm_int)

        bias = 1. + (self.delta_c ** 2. - sigma_sq_int) /(sigma_sq_int * self.delta_c)
        winf = self.windowf((10.**self.lgkh_int[None, None, :])*r_mass[:, :, None])

        b0 = np.sum( lgm_wet[None, :, None]\
                * 10.**(2.*lgm_int[None, :, None]) * np.log(10.)\
                * dndm_int[:, :, None] * winf**2., 
                axis=1)
        b1 = np.sum( lgm_wet[None, :, None]\
                * 10.**(2.*lgm_int[None, :, None]) * np.log(10.)\
                * dndm_int[:, :, None] * bias[:, :, None] * winf**2., 
                axis=1)
        b2 = np.sum( lgm_wet[None, :, None]\
                * 10.**(2.*lgm_int[None, :, None]) * np.log(10.)\
                * dndm_int[:, :, None] * bias[:, :, None]**2 * winf**2., 
                axis=1)

        b1 /= b0
        b2 /= b0

        return b1, b2


    #@utils.log_timing
    def b_halo(self, dndm=None, sigma_sq=None, mass=None, 
            logmass_min=10, logmass_max=16):

        #print "Estimate Halo Bias"

        if dndm is None:
            mass = np.logspace(logmass_min, logmass_max, 100)
            dndm = self.dn_dmass(mass)
            sigma_sq = self.sigma_sq
        dndm_f = lambda lgm, i: np.interp(lgm, np.log10(mass), dndm[i,:])
        #dndm_f = interpolate.interp1d(np.log10(mass), dndm)

        #bias = 1. + (self.delta_c ** 2. - sigma_sq[:, :, None])\
        #        /(sigma_sq[:, :, None] * self.delta_c * self.da[:, None, :])
        #bias_f = lambda lgm, k, i: \
        #        interpolate.interp2d(np.log10(mass), self.kh, bias[i,...].T)\
        #        (lgm, k)
        bias = 1. + (self.delta_c ** 2. - sigma_sq) /(sigma_sq * self.delta_c)
        bias_f = lambda lgm, i: np.interp(lgm, np.log10(mass), bias[i,:])

        itf = lambda lgm, k, i, q: \
                10.** (2.*lgm) * dndm_f(lgm, i)\
                * bias_f(lgm, i) ** q
                #* self.windowf(k * self.r_func(10.**lgm)[i]) ** 2.
        b_f = np.vectorize(lambda k, i, q: \
            fixed_quad(itf, logmass_min, logmass_max, args=(k, i, q), n=10000)[0])

        itf0 = lambda lgm, k, i: 10.** (2.*lgm) * dndm_f(lgm, i)
                #* self.windowf(k * self.r_func(10.**lgm)[i]) ** 2.
        b_f0 = np.vectorize(lambda k, i: \
            fixed_quad(itf0, logmass_min, logmass_max, args=(k, i), n=10000)[0])

        #ks = np.array([0.0001, 0.01, 0.1, 1, 10., 20., 30., 100.])
        #kh = self.kh.flatten()
        kh = 10.**self.lgkh_int
        #ks = self.kh.flatten()
        b0 = np.zeros([len(self.z), len(kh)])
        b1 = np.zeros([len(self.z), len(kh)])
        b2 = np.zeros([len(self.z), len(kh)])

        ks = np.logspace(np.log10(kh.min()), np.log10(kh.max()), 200)
        #ks = np.linspace(kh.min(), kh.max(), 20)
        for i in range(len(self.z)):
            #print "z =  %s"% self.z[i]
            #for j in range(len(ks)):

            b0[i, :] = np.interp(np.log10(kh), np.log10(ks), b_f0(ks, i))
            b1[i, :] = np.interp(np.log10(kh), np.log10(ks), b_f(ks, i, 1))
            b2[i, :] = np.interp(np.log10(kh), np.log10(ks), b_f(ks, i, 2))

            #b0[i, :] = b_f0(ks, i)
            #b1[i, :] = b_f(ks, i, 1)
            #b2[i, :] = b_f(ks, i, 2)

            #b0[i, :] = interpolate.interp1d(ks, b_f0(ks, i))(kh)
            #b1[i, :] = interpolate.interp1d(ks, b_f(ks, i, 1))(kh)
            #b2[i, :] = interpolate.interp1d(ks, b_f(ks, i, 2))(kh)

        b1 /= b0
        b2 /= b0

        return b1, b2, kh


class Tinker08(HALO_MASS_FUNCTION):

    def __init__(self, *args, **kwargs):

        super(Tinker08, self).__init__(*args, **kwargs)

        self.Delta=200.

    def f_sigma(self, sigma_sq, A=1.858659e-01, a=1.466904e+00, 
            b=2.571104e+00, c=1.193958e+00):

        A = A * (1. + self.z) ** (-0.14)
        a = a * (1. + self.z) ** (-0.06)
        alpha = 10. ** ( - (0.75/np.log10(self.Delta/75.)) ** 1.2 )
        b = b * (1. + self.z) ** -alpha

        return A * ( (np.sqrt(sigma_sq)/b)**(-a) + 1. ) * np.exp(-c/sigma_sq)

class Tinker10(HALO_MASS_FUNCTION):

    def __init__(self, *args, **kwargs):

        super(Tinker10, self).__init__(*args, **kwargs)

        self.Delta=200.

    def f_sigma(self, sigma_sq, alpha=0.368, beta=0.589, gamma=0.864, 
            phi=-0.729, eta=-0.243):

        nu = self.delta_c / np.sqrt(sigma_sq)

        beta  = beta  * (1. + self.z) **  0.20
        phi   = phi   * (1. + self.z) ** -0.08
        eta   = eta   * (1. + self.z) **  0.27
        gamma = gamma * (1. + self.z) ** -0.01

        return alpha * (1. + (beta * nu) ** (-2. * phi)) * (nu ** (2. * eta)) \
                * np.exp(-gamma * nu ** 2. / 2.) * nu

class Bhattacharya(HALO_MASS_FUNCTION):
    '''
    Updated Sheth & Tormen mass function, 
    see [Suman Bhattacharya ApJ 732:122 (2011)]
    '''

    def __init__(self, *args, **kwargs):

        super(Bhattacharya, self).__init__(*args, **kwargs)

        self.Delta=200.

    def f_sigma(self, sigma_sq, A0=0.333, a0=0.788, p0=0.807, q0=1.795):

        nu = self.delta_c / np.sqrt(sigma_sq)

        A = A0 / (1. + self.z) ** 0.11
        a = a0 / (1. + self.z) ** 0.01
        p = p0
        q = q0

        return A * (2./np.pi)**0.5 * np.exp(- a * nu**2. / 2.) \
                * (1. + (1./(a*nu**2))**p) * (nu*a**0.5)**q


def growth_rate_fitting(kh, redshifts, omm=0.27, omnu=0, w=-1, kind='slinear'):
    '''
    Angeliki Kiakotou et. al. PRD 77, 063005 (2008)
    '''

    redshifts = np.array(redshifts)

    kh_fitting = np.array([0.001, 0.01, 0.05, 0.07, 0.1, 0.5])
    A_fitting = np.array([0.0, 0.132, 0.613, 0.733, 0.786, 0.813])
    B_fitting = np.array([0.0, 1.620, 5.590, 6.000, 5.090, 0.803])
    C_fitting = np.array([0.0, 7.130, 21.13, 21.45, 15.50, -0.844])

    A = interpolate.interp1d(np.log10(kh_fitting), A_fitting, kind=kind)
    B = interpolate.interp1d(np.log10(kh_fitting), B_fitting, kind=kind)
    C = interpolate.interp1d(np.log10(kh_fitting), C_fitting, kind=kind)

    fnu = omnu / omm
    alpha0 = 3. / (5. - w/(1.-w))
    alpha1 = 3. / 125. * ( ((1.-w)*(1.-3.*w/2.))/(1.-6.*w/5.)**3 )
    alpha  = alpha0 + alpha1 * (1. - omm)

    mu0 = lambda kh: 1.
    mu1 = lambda kh: 1. - (1.-omm) * A(np.log10(kh)) * fnu \
            + B(np.log10(kh)) * fnu**2. - C(np.log10(kh)) * fnu**3.
    mu2 = lambda kh: (1. - fnu) ** alpha0

    mu = lambda kh: np.piecewise(kh, 
            [kh<0.001, np.logical_and(kh>=0.001, kh<=0.5), kh>0.5], [mu0, mu1, mu2])

    #plot.plot_fitting_ABC(kh, A(np.log10(kh)), B(np.log10(kh)), C(np.log10(kh)))
    #plot.plot_fitting_f(kh, mu(kh))
    Ea = omm * (1. + redshifts)**3. + (1. - omm) * (1. + redshifts)**(3.*(1.+w))
    #return mu(kh)[None, :] * ((omm * (1.+redshifts)**3/Ea)**alpha)[:, None], mu(kh)
    return (omm * (1.+redshifts)**3/Ea)**alpha, mu(kh)

def mean_pairwise_v_fitting(CAMB, pk, kh, Ea, omnu, rh=None, rmin=10, rmax=200, 
        logmass_min=10, logmass_max=16, redshifts=[0,]):
    t1 = time.time()

    redshifts = np.array(redshifts)

    c = CAMB
    omm = (float(c.params['ombh2']) + float(c.params['omch2']))\
            / (float(c.params['hubble'])/100.)**2.

    f, mu = growth_rate_fitting(kh, redshifts, omm=omm, omnu=omnu, kind='quadratic')
    pk    = pk * mu[None, :]**2.
    hmf = Tinker10(pk, kh, redshifts, omm=omm, npoints=10000)
    #hmf = Bhattacharya(pk, kh, redshifts, omm=omm, npoints=10000)
    b1, b2, kh = hmf.b_halo(logmass_min=logmass_min, logmass_max=logmass_max)
    pk = hmf.pk_int
    pk_nu = pk * f[:, None]# * mu[None, :]
    b1_f = lambda k, i: np.interp(k, kh, b1[i,:])
    #b2_f = lambda k, i: np.interp(k, kh, b2[i,:])

    xi, rh = c.matter_correlation_function_fixed(r=rh, rmin=rmin, rmax=rmax, 
            pk=pk, kh=kh, khw = hmf.lgkh_wet, redshifts=redshifts, bias_f=b2)
    xi_bar, rh_bar = c.mean_matter_correlation_function(r=rh, rmin=rmin, rmax=rmax, pk=pk_nu, kh=kh, 
            redshifts=redshifts, bias_f=b1_f)

    #vij = - 2./3. * Ea[:,None] * (1./(1.+redshifts[:,None])) * fa[:,None] \
    #        * rh[None, :] * xi_bar / (1. + xi)
    vij = - 2./3. * Ea[:,None] * (1./(1.+redshifts[:,None])) * rh[None, :]\
            * xi_bar / (1. + xi)
    print "[TIMING] Estimate Pairwise Velocity: %8.4f [s]"%(time.time() - t1)
    return vij, rh



def mean_pairwise_v(CAMB, rh=None, rmin=10, rmax=200, logmass_min=10, logmass_max=16):
    #t1 = time.time()

    c = CAMB

    redshifts = c.redshifts
    redshifts_plus = c.redshifts_plus
    sort_order = c.redshifts_sort_order

    omm = (float(c.params['ombh2']) + float(c.params['omch2']))\
            / (float(c.params['hubble'])/100.)**2.
    hmf = Tinker10(redshifts=redshifts_plus, omm=omm, npoints=10000)
    hmf.get_fixed_kh(kmin=1.e-4, kmax=3.e1)
    kh = 10.**hmf.lgkh_int

    pk_sort, kh = c.matter_power_spectrum_2d1d(kh = kh, npoints=500)
    Ea = c.hubble_parameter(redshifts) / float(c.params['hubble']) * 100.
    #t1 = time.time()
    #fa = c.lin_growth_rate(redshifts)
    pk = np.zeros_like(pk_sort)
    pk[sort_order,:] = pk_sort

    hmf.pk_int = pk
    b1, b2, kh = hmf.b_halo(logmass_min=logmass_min, logmass_max=logmass_max,)

    pk_plus = hmf.pk_int[len(redshifts):, :]
    pk = hmf.pk_int[:len(redshifts), :]
    b1_plus = b1[len(redshifts):, :]
    b1 = b1[:len(redshifts), :]
    b2_plus = b2[len(redshifts):, :]
    b2 = b2[:len(redshifts), :]

    dpkda = - (1. + redshifts[:, None]) * (pk_plus*b1_plus - pk*b1)\
            / (redshifts[:,None] * 0.1)

    xi, rh = c.matter_correlation_function_fixed(r=rh, rmin=rmin, rmax=rmax, 
            pk=pk, kh=kh, khw = hmf.lgkh_wet, redshifts=redshifts, bias_f=b2)
    xi_bar, rh_bar = c.mean_matter_correlation_function(r=rh, rmin=rmin, rmax=rmax, 
            pk=dpkda, kh=kh, redshifts=redshifts, bias_f=None)

    vij = - 1./3. * Ea[:,None] * (1./(1.+redshifts[:,None])) * rh[None, :]\
            * xi_bar / (1. + xi)
    #print "[TIMING] Estimate Pairwise Velocity: %8.4f [s]"%(time.time() - t1)
    return vij, rh

def velocity_mnu(output_path='./', output_name = 'vij.h5', logmass_min=10, 
        z=0., fitting=False):

    redshifts = np.array([z, ])
    c = camb_wrap.CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')
    mnu_list = np.linspace(0, 0.6, 4)
    #mnu_list = [0.0]

    vij_list = []
    #for mnu in mnu_list:
    if fitting:
        c.params['omnuh2'] = 0.
        c.set_cosmology_params()
        c.get_results()
        Ea = c.hubble_parameter(redshifts) / float(c.params['hubble']) * 100.
        print c.hubble_parameter(0)
        pk, kh = c.matter_power_spectrum(redshifts=redshifts, 
                kmin=1.e-4, kmax=3.e2, npoints=5000)
        for mnu in mnu_list:
            vij, rh = mean_pairwise_v_fitting(c, pk, kh, Ea, 
                mnu/94.07/(float(c.params['hubble'])/100.)**2., 
                redshifts=redshifts, logmass_min=logmass_min)
            vij_list.append(vij)
    else:
        for mnu in mnu_list:
            c.params['omnuh2'] = mnu / 94.07
            c.set_cosmology_params()
            c.set_matter_power(redshifts=redshifts, kmax=3.e2)
            #c.get_results()
            vij, rh = mean_pairwise_v(c, logmass_min=logmass_min)
            vij_list.append(vij)

    vij_list = np.array(vij_list)
    vij_results = h5py.File(output_path + output_name, 'w')
    vij_results['vij'] = vij_list
    vij_results['rh']  = rh
    vij_results['nu']  = mnu_list
    vij_results['zs']  = redshifts
    vij_results.close()



def stellarM2haloM(stellar_mass):

    '''
    According to the formular of Behroozi et. al. ApJ 717,379 (2010)
    and the fitting parameters from Alexie Leauthaud et. al. ApJ 744,159 (2012)
    of redshfit bin z1 = [0.22, 0.48]
    '''

    log10M1     = 12.520
    log10Mstar0 = 10.916
    beta        = 0.457
    delta       = 0.566
    gamma       = 1.53

    Mstar_r = stellar_mass / (10.**log10Mstar0)

    log10Mh = log10M1 + beta * (np.log10(stellar_mass) - log10Mstar0) \
            + Mstar_r ** delta / (1. + Mstar_r ** (-gamma)) - 0.5


    return 10.**log10Mh

if __name__=="__main__":

    ##kh = np.linspace(0.001, 10, 500)
    #kh = np.logspace(-4, 2, 500)
    #redshifts = [0., 0.1, 0.2]
    #fnu_list = [0, 0.01, 0.08, 0.15, 0.2]
    #f = np.zeros([len(fnu_list),len(redshifts),len(kh)])
    #for i, fnu in enumerate(fnu_list):
    #    f[i] = growth_rate_fitting(kh, redshifts, omm=0.3, omnu=fnu*0.3, 
    #            kind='cubic')
    #            #kind='quadratic')

    #plot.plot_fitting_f(kh, f, fnu_list, redshifts)

    #plt.show()

    #exit()


    fitting = False
    logmass_min = 14
    z = 0.05
    output_name='new_vij_%smnu_Mmin%d_nowindow_z%3.2f.h5'
    if fitting:
        output_name = output_name%('fitting_', logmass_min, z)
    else:
        output_name = output_name%('', logmass_min, z)
    velocity_mnu(output_path='./data/', output_name=output_name, 
            z=z, logmass_min=logmass_min, fitting=fitting)
    exit()



    c = camb_wrap.CAMB(ini_param_file='./data/params.ini')
    redshifts = np.array([2., 1., 0.2, 0])
    mass = np.logspace(10, 15, 100)
    pk, kh = c.matter_power_spectrum(redshifts=redshifts, 
            kmin=1.e-4, kmax=3.e2, npoints=5000)
    #plot.plot_PK(pk, kh, redshifts)

    #xi, rh = c.matter_correlation_function(pk=pk, kh=kh, redshifts=redshifts)
    #plot.plot_XI(xi, rh, redshifts)
    #xi_bar, rh_bar = c.mean_matter_correlation_function(pk=pk, kh=kh, redshifts=redshifts)
    #plot.plot_XI(xi_bar, rh_bar, redshifts, volume_averaged=True)

    #da, fa = c.growth_factor(kh=kh, redshifts=redshifts)
    #plot.plot_growth(da, fa, kh, redshifts=redshifts)

    omm = (float(c.params['ombh2']) + float(c.params['omch2']))\
            / (float(c.params['hubble'])/100.)**2.
    #hmf = Tinker08(pk, kh, redshifts, omm=omm)
    hmf = Tinker10(pk, kh, redshifts, omm=omm)
    #plot.plot_PK(pk, kh, redshifts, hmf.pk_int, hmf.lgkh_int)
    #sig_sq = hmf.sigma_square(mass)
    #sig_sq_old = hmf.sigma_square_old(mass)
    #plot.plot_sigma_sq(mass, sig_sq, redshifts, sig_sq_old)
    b1, b2, kh = hmf.b_halo()
    #b1_old, b2_old = hmf.b_halo_old()
    #plot.plot_bias(b1, b2, 10.**hmf.lgkh_int, redshifts, b1_old, b2_old)
    #plt.show()
    exit()
    #dndm = hmf.dn_dmass(mass)
    #plot.plot_dndm(dndm, mass, redshifts,  rhom_mean = hmf.rhom_mean,
    #        sigma_sq=hmf.sigma_sq, output='../png/')
    #plot.plot_f_sigma(hmf.f_sigma(hmf.sigma_sq), hmf.sigma_sq, redshifts, 
    #        output='../png/')

    b1, b2, kh = hmf.b_halo()
    plot.plot_bias(b1, b2, kh, redshifts)
    b1 = np.concatenate([kh[None, :], b1], axis=0)
    b2 = np.concatenate([kh[None, :], b2], axis=0)
    redshifts = np.concatenate([[-99, ], redshifts])
    b1 = np.concatenate([redshifts[:,None], b1], axis=1)
    b2 = np.concatenate([redshifts[:,None], b2], axis=1)
    np.savetxt('./data/b1.txt', b1, fmt='%8.5e')
    np.savetxt('./data/b2.txt', b2, fmt='%8.5e')

    plt.show()









