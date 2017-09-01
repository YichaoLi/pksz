import numpy as np
import utils
import units
from scipy.special import spherical_jn as jn
from scipy.special import p_roots
from scipy.integrate import fixed_quad
from scipy import interpolate
import camb
from camb import model
import iniFile


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

class CAMB(iniFile.iniFile):

    def __init__(self, ini_param_file = None, ini_params = None):

        super(CAMB, self).__init__()
        if ini_param_file is not None:
            self.readFile(ini_param_file)
        if ini_params is not None:
            self.initialize_params(ini_params)

        self.camb_params = camb.CAMBparams()
        self.set_cosmology_params()
        self.set_initpower_params()
        self.set_darkenergy_params()
        self.set_other_params()

        self.results = None

        self.Hz_fid = None
        self.Da_fid = None

    #@utils.log_timing
    def get_results(self):
        self.results = camb.get_results(self.camb_params)

    #@utils.log_timing
    def set_cosmology_params(self):
        self.camb_params.set_cosmology(
                H0                    = float(self.params['hubble']),
                cosmomc_theta         = None,
                ombh2                 = float(self.params['ombh2']), 
                omch2                 = float(self.params['omch2']), 
                omk                   = float(self.params['omk']), 
                neutrino_hierarchy    = 'degenerate', 
                num_massive_neutrinos = int(self.params['massive_neutrinos']), 
                mnu                   = 94.07 * float(self.params['omnuh2']), 
                nnu                   = float(self.params['massless_neutrinos']) \
                                      + float(self.params['massive_neutrinos']),
                YHe                   = float(self.params['helium_fraction']), 
                meffsterile           = 0, 
                standard_neutrino_neff= 3.046, 
                TCMB                  = 2.7255, 
                tau                   = float(self.params['re_optical_depth']), 
                #tau_neutron           = 880.3
                )
        self.results = None
    #@utils.log_timing
    def set_initpower_params(self):
        self.camb_params.InitPower.set_params(
                As                    = float(self.params['scalar_amp(1)']),
                ns                    = float(self.params['scalar_spectral_index(1)']),
                nrun                  = float(self.params['scalar_nrun(1)']),
                nrunrun               = float(self.params['scalar_nrunrun(1)']),
                r                     = float(self.params['initial_ratio(1)']),
                nt                    = float(self.params['tensor_spectral_index(1)']),
                ntrun                 = float(self.params['tensor_nrun(1)']),
                pivot_scalar          = float(self.params['pivot_scalar']),
                pivot_tensor          = float(self.params['pivot_tensor']),
                parameterization      = int(self.params['tensor_parameterization']))
        self.results = None
    #@utils.log_timing
    def set_darkenergy_params(self):
        self.camb_params.set_dark_energy(
                w                     = float(self.params['w']), 
                sound_speed           = float(self.params['cs2_lam']), 
                dark_energy_model     = 'fluid')
                #wa                    = float(self.params['wa']),
                #dark_energy_model     = 'ppf')
        self.results = None
    #@utils.log_timing
    def set_other_params(self):
        self.camb_params.WantTensors = int(self.params['get_tensor_cls'])
        self.camb_params.set_for_lmax(
                lmax                  = int(self.params['l_max_scalar']), 
                lens_potential_accuracy = 0)
        self.camb_params.NonLinear = int(self.params['do_nonlinear'])
        self.camb_params.DoLensing = int(self.params['do_lensing'])
        #self.camb_params.set_accuracy(
        #        AccuracyBoost         = float(self.params['accuracy_boost']), 
        #        lSampleBoost          = float(self.params['l_sample_boost']), 
        #        lAccuracyBoost        = float(self.params['l_accuracy_boost']), 
        #        HighAccuracyDefault   =True, 
        #        DoLateRadTruncation   =True
        #        )
        #self.camb_params.set_bbn_helium(
        #        ombh2                 = float(self.params['ombh2']),
        #        delta_nnu             = float(self.params['massless_neutrinos']) \
        #                              + float(self.params['massive_neutrinos']) \
        #                              - 3.046,
        #        tau_neutron           = 880.3,)
        self.results = None

    def cmb_angular_power_spectrum(self):

        if self.results is None:
            self.get_results()

        powers = self.results.get_cmb_power_spectra(self.camb_params)

        totCl = powers['total']
        ls = np.arange(totCl.shape[0])
        #totCl = totCl[2:, :]
        #ls = ls[2:]

        return totCl[:, 0], ls
        #print totCl.shape

    #def k_samples(self, kmin=1.e-4, kmax=2, npoints=2000):
    #    x, khw = _cached_roots_legendre(npoints)  #p_roots(npoints)
    #    kh = (kmax - kmin) * (x + 1.) / 2.0 + kmin
    #    return kh, khw * (kmax - kmin) / 2.0

    def get_fid_Hz_Daz(self, z):
        '''
        We use w=-1, wa=0 as the fiducial cosmology
        '''

        w0_tmp = float(self.params['w'])
        wa_tmp = float(self.params['wa'])

        self.params['w'] = -1
        self.params['wa'] = 0

        self.Hz_fid = self.hubble_parameter(z)
        self.Da_fid = self.angular_distance(z)

        self.params['w'] = w0_tmp
        self.params['wa'] = wa_tmp

    def set_matter_power(self, redshifts=[0, ], kmax=2.):

        redshifts = np.array(redshifts)
        redshifts_plus = np.concatenate([redshifts, redshifts+redshifts*0.1])
        sort_order = np.argsort(redshifts_plus)[::-1]

        self.redshifts = redshifts
        self.redshifts_plus = redshifts_plus
        self.redshifts_sort_order = sort_order

        self.camb_params.WantTransfer = True
        self.camb_params.set_matter_power(redshifts=redshifts_plus[sort_order], 
                kmax=kmax/0.7*2.)
        self.get_results()


    #@utils.log_timing
    def matter_power_spectrum_2d1d(self, kh=None, kmin=1.e-4, kmax=10., npoints=500,
            rsd=False):

        if kh is not None:
            kmin = kh.min()
            kmax = kh.max()

        _kh, z, pk = self.results.get_matter_power_spectrum(
                minkh=kmin/10., maxkh=kmax*1.5, npoints = npoints, 
                have_power_spectra=True)

        if kh is None:
            kh = _kh[np.logical_and(_kh<kmax, _kh>kmin)]

        z = np.array(z[::-1])
        pk = pk[::-1, :]
        
        self.get_fid_Hz_Daz(z)
        Hz = self.hubble_parameter(z)
        Da = self.angular_distance(z)

        r_para_ratio = (self.Hz_fid / Hz)
        r_perp_ratio = (Da / self.Da_fid)
        
        v_ratio = r_perp_ratio ** 2 * r_para_ratio

        #print r_para_ratio
        #print r_perp_ratio
        #print v_ratio

        # function that convert k_para and k_perp to fiducial model
        k_para = lambda k, mu, i: k[None, :] * mu[:, None] * r_para_ratio[i]
        k_perp = lambda k, mu, i: k[None, :] * (1. - mu[:, None]**2)**0.5\
                * r_perp_ratio[i]
        k_tot  = lambda k, mu, i: (k_para(k, mu, i)**2 + k_perp(k, mu, i)**2)**0.5

        if rsd:
            h = float(self.params['hubble']) / 100.
            omm = (float(self.params['ombh2']) + float(self.params['omch2'])) / h**2
            beta = (omm * (1.+z)**3) ** 0.6
        else:
            beta = np.zeros(z.shape)

        # for fixed_quad intigration
        x, w = _cached_roots_legendre(npoints)
        mu = (x + 1.)/2.
        w = 0.5 * w

        pk1d = np.zeros(z.shape + kh.shape)
        for i in range(len(z)):

            pk1d_f = interpolate.interp1d(np.log10(_kh), pk[i, :])

            # return matrix in shape of mu x kh
            pk1d[i, :] = np.sum(w[:, None] * (v_ratio[i]\
                    * ((1. + beta[i] * k_para(kh, mu, i)**2/k_tot(kh, mu, i)**2)**2) \
                    * pk1d_f(np.log10(k_tot(kh, mu, i)))), axis=0)

            #pk2d = lambda mu, k: v_ratio[i]\
            #        * ((1. + beta[i] * k_para(k, mu, i)**2/k_tot(k, mu, i)**2)**2) \
            #        * pk1d_f(np.log10(k_tot(k, mu, i)))
            #pk1d[i, :] = 2. * fixed_quad(pk2d, 0, 1, args=(kh,), n=1000)

        return pk1d, kh


    #@utils.log_timing
    def matter_power_spectrum(self, redshifts=[0.,], kmin=1.e-4, kmax=2, npoints=2000):

        #print "Estimate PK"

        #if self.results is None:
        #    self.get_results()

        self.camb_params.WantTransfer = True
        self.camb_params.set_matter_power(redshifts=redshifts, kmax=kmax/0.7*1.5)
        self.get_results()

        kh, z, pk = self.results.get_matter_power_spectrum(
                minkh=kmin, maxkh=kmax, npoints = npoints, have_power_spectra=True)
        #kh, z, pk = self.results.get_linear_matter_power_spectrum(
        #        have_power_spectra=True)

        z = z[::-1]
        pk = pk[::-1, :]

        return pk, kh
        #s8 = np.array(self.results.get_sigma8())
        #return pk, kh, s8

    #@utils.log_timing
    def mean_matter_correlation_function(self, r=None, rmin=10, rmax=200, 
            pk=None, kh=None, redshifts=[0,], npoints=200, bias_f=None):

        #print "Estimate Volume-Averaged Correlation function"

        if pk is None:
            self.camb_params.WantTransfer = True
            self.camb_params.set_matter_power(redshifts=redshifts, kmax=50.0)
            self.get_results()

            kh, z, pk = self.results.get_matter_power_spectrum(
                minkh=1.e-4, maxkh=50., npoints = npoints)
        #npoints = pk.shape[1]
        if r is None:
            r = np.linspace(rmin, rmax, npoints)
        xi = np.zeros([len(redshifts), r.shape[0]])
        r_tmp = np.linspace(0, rmax, npoints)

        k_min = kh.min()
        k_max = kh.max()

        if bias_f is not None:
            pkf = lambda k, i: np.interp(k, kh, pk[i]) * bias_f(k, i)
        else:
            pkf = lambda k, i: np.interp(k, kh, pk[i])
        itf = lambda k, r, i: k**2 * pkf(k, i) * jn(0, k*r) / 2. / np.pi**2
        xif = np.vectorize(lambda r, i: \
                fixed_quad(itf, k_min, k_max, args=(r, i), n=10000)[0])
        #xibf = np.vectorize(lambda r, i: 3. / r**3.\
        #        * fixed_quad(xif, 0, r, args=(i), n=10000)[0])
        for i in range(len(redshifts)):
            #print "z = %3.1f"%redshifts[i]
            xi_tmp = r_tmp**2 * xif(r_tmp, i)
            itf2 = lambda x: np.interp(x, r_tmp, xi_tmp)
            xibf = np.vectorize(lambda x: fixed_quad(itf2, 0, x, n=10000)[0])
            xi[i,:] = 3./r**3. * xibf(r)
            #xi[i,:] = 1./r**2. * xibf(r)

        return xi, r

    #@utils.log_timing
    def matter_correlation_function_fixed(self, r=None, rmin=10, rmax=200, 
            pk=None, kh=None, khw=None, redshifts=[0,], npoints=200, bias_f=None):

        if bias_f is None:
            bias_f = np.ones(pk)

        if r is None:
            r = np.linspace(rmin, rmax, npoints)[None, None, :]
        xi = np.zeros([len(redshifts), r.shape[0]])

        kh = kh[None, :, None]
        xi = np.sum( khw[None, :, None] * kh**3 * np.log(10.) * jn(0, kh*r)\
                * pk[:, :, None] * bias_f[:, :, None], axis=1) / 2. / np.pi**2

        ##lgk_min = np.log10(kh.min())
        ##lgk_max = np.log10(kh.max())
        ##pkf = lambda lgk, i: np.interp(lgk, np.log10(kh), pk[i])
        ##itf = lambda lgk, r, i: 10.**(lgk*3) * np.log(10.) * pkf(lgk, i)\
        ##        * jn(0, (10.**lgk)*r) / 2. / np.pi**2
        ##xif = np.vectorize(lambda r, i: \
        ##        fixed_quad(itf, lgk_min, lgk_max, args=(r, i), n=10000)[0])
        #if bias_f is not None:
        #    pkf = lambda k, i: np.interp(k, kh, pk[i]) * bias_f(k, i)
        #else:
        #    pkf = lambda k, i: np.interp(k, kh, pk[i])
        #itf = lambda k, r, i: k**2 * pkf(k, i) * jn(0, k*r) / 2. / np.pi**2
        #xif = np.vectorize(lambda r, i: \
        #        fixed_quad(itf, k_min, k_max, args=(r, i), n=10000)[0])

        #for i in range(len(redshifts)):
        #    #print "z = %3.1f"%redshifts[i]
        #    xi[i,:] = xif(r, i)

        return xi, r.flatten()

    #@utils.log_timing
    def matter_correlation_function(self, r=None, rmin=10, rmax=200, pk=None, 
            kh=None, redshifts=[0,], npoints=200, bias_f=None):

        #print "Estimate Correlation function"

        if pk is None:
            self.camb_params.WantTransfer = True
            self.camb_params.set_matter_power(redshifts=redshifts, kmax=50.0)
            self.get_results()

            kh, z, pk = self.results.get_matter_power_spectrum(
                minkh=1.e-4, maxkh=50., npoints = npoints)
        #npoints = pk.shape[1]
        if r is None:
            r = np.linspace(rmin, rmax, npoints)
        xi = np.zeros([len(redshifts), r.shape[0]])

        k_min = kh.min()
        k_max = kh.max()

        #lgk_min = np.log10(kh.min())
        #lgk_max = np.log10(kh.max())
        #pkf = lambda lgk, i: np.interp(lgk, np.log10(kh), pk[i])
        #itf = lambda lgk, r, i: 10.**(lgk*3) * np.log(10.) * pkf(lgk, i)\
        #        * jn(0, (10.**lgk)*r) / 2. / np.pi**2
        #xif = np.vectorize(lambda r, i: \
        #        fixed_quad(itf, lgk_min, lgk_max, args=(r, i), n=10000)[0])
        if bias_f is not None:
            pkf = lambda k, i: np.interp(k, kh, pk[i]) * bias_f(k, i)
        else:
            pkf = lambda k, i: np.interp(k, kh, pk[i])
        itf = lambda k, r, i: k**2 * pkf(k, i) * jn(0, k*r) / 2. / np.pi**2
        xif = np.vectorize(lambda r, i: \
                fixed_quad(itf, k_min, k_max, args=(r, i), n=20000)[0])

        for i in range(len(redshifts)):
            #print "z = %3.1f"%redshifts[i]
            xi[i,:] = xif(r, i)

        return xi, r

    #@utils.log_timing
    def hubble_parameter(self, redshifts):
        if self.results is None:
            self.get_results()

        H0 = float(self.params['hubble'])
        z = redshifts

        omm = (float(self.params['ombh2']) + float(self.params['omch2'])) \
                / (float(self.params['hubble'])/100.)**2.
        omk = float(self.params['omk'])
        oml = 1. - omm - omk

        w0 = float(self.params['w'])
        wa = float(self.params['wa'])
        #wz = w0 + wa * (z/(1. + z))

        omm_z = omm * (1.+z)**3
        omk_z = omk * (1.+z)**2
        oml_z = oml * np.exp(3. * ((1. + w0 + wa) * np.log(1.+z) - wa * z / (1.+z)))

        #hubble_parameter = np.vectorize(self.results.hubble_parameter)
        #y = hubble_parameter(redshifts)
        #y2= H0 * (omm_z + omk_z + oml_z) ** 0.5

        return H0 * (omm_z + omk_z + oml_z) ** 0.5


    #@utils.log_timing
    def lin_growth_rate(self, redshifts, zmin=0., zmax=3., znum=100):
        if self.results is None:
            self.get_results()


        #print "Estimate Growth Factor"
        #zs = np.linspace(zmin, zmax, znum)
        ks = [0.001, ]
        return self.results.get_redshift_evolution(ks, redshifts, ['growth',])[0,:,0]

        #print fa.shape
        #sort_order = np.argsort(redshifts)
        #fa_result = np.zeros((len(redshifts)))
        #fa_result[sort_order] = np.interp(redshifts[sort_order], zs, fa)
        #return fa_result


    def growth_factor(self, kh, redshifts, zmin=0, zmax=3., znum=100, full=False):

        #print "Estimate Growth Factor"
        if self.results is None:
            self.get_results()


        zs = np.linspace(zmin, zmax, znum)
        ks = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100, 300]

        fa = self.results.get_redshift_evolution(ks, zs, ['growth','a'])
        a  = fa[:,:,1]
        fa = fa[:,:,0]
        da = fa[:,:-1] * (np.log(a[:,1:]) - np.log(a[:,:-1]))
        da = np.concatenate([np.zeros(len(ks))[:,None], da], axis=1)
        da = np.exp(np.cumsum(da, axis=1))

        sort_order = np.argsort(redshifts)

        fa_interp = interpolate.interp2d(zs, ks, fa, kind='cubic', bounds_error=True)
        da_interp = interpolate.interp2d(zs, ks, da, kind='linear', bounds_error=True)

        fa_result = np.zeros((len(kh), len(redshifts)))
        da_result = np.zeros((len(kh), len(redshifts)))
        fa_result[:,sort_order] = fa_interp(redshifts[sort_order], kh)
        da_result[:,sort_order] = da_interp(redshifts[sort_order], kh)

        if full:
            return da_result, fa_result, da, fa, zs, ks
        else:
            return da_result.T, fa_result.T
            

    def initialize_params(self, ini_params):

        modified_keys = []
        ignored_keys = ['redshift_sigma_list', 'redshift_bias_list', 
                        'redshift_kind_list']
        for key in ini_params.keys():
            if key in self.params.keys():
                modified_keys += [key, ]
                self.params[key] = ini_params[key]
            elif key == 'redshift_list':
                redshift_list = ini_params['redshift_list']
                self.redshift_list = redshift_list
                redshift_numb = len(redshift_list)
                self.params['num_redshiftwindows'] = redshift_numb
                redshift_sigma_list = ini_params['redshift_sigma_list']
                redshift_bias_list  = ini_params['redshift_bias_list']
                redshift_kind_list  = ini_params['redshift_kind_list']
                if redshift_sigma_list == []:
                     redshift_sigma_list = \
                             [self.params['redshift_sigma(1)'], ] * redshift_numb
                if redshift_bias_list == []:
                     redshift_bias_list = \
                             [self.params['redshift_bias(1)'],  ] * redshift_numb
                if redshift_kind_list == []:
                     redshift_kind_list = \
                             [self.params['redshift_kind(1)'],  ] * redshift_numb
                for i in range(redshift_numb):
                    self.params['redshift(%d)'%(i+1)] = redshift_list[i]
                    self.params['redshift_sigma(%d)'%(i+1)] = redshift_sigma_list[i]
                    self.params['redshift_bias(%d)'%(i+1)] = redshift_bias_list[i]
                    self.params['redshift_kind(%d)'%(i+1)] = redshift_kind_list[i]
                    modified_keys += ['redshift(%d)'%(i+1), ]
                    modified_keys += ['redshift_sigma(%d)'%(i+1), ]
                    modified_keys += ['redshift_bias(%d)'%(i+1), ]
                    modified_keys += ['redshift_kind(%d)'%(i+1), ]

            elif key in ignored_keys :
                continue
            else:
                raise "Key Error: [%s] "%key

    def comoving_distance(self, z):

        '''
        in unit of Mpc/h

        '''

        h = float(self.params['hubble']) / 100.

        f = lambda z1: units.c / self.hubble_parameter(z1) / 1.e3

        return _intfz(f, 0.0, z) * h

    def proper_distance(self, z):

        x = self.comoving_distance(z)

        omk = float(self.params['omk'])

        dhi = np.sqrt(np.fabs(omk)) * 100. / (units.c / 1.e3)

        if omk < 0.:
            x = np.sin(x*dhi) / dhi
        elif omk > 0.:
            x = np.sinh(x*dhi) / dhi

        return x

    def angular_distance(self, z):

        return self.proper_distance(z) / (1.+z)

    def luminosity_distance(self, z):

        return self.proper_distance(z) * (1.+z)

#_camb = CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')

@np.vectorize
def _intfz(f, a, b, args=()):
    # A wrapper function to allow vectorizing integrals, cuts such
    # that integrals to very high-z converge (by integrating in log z
    # instead).
    #import integrate

    def _int(f, a, b):
        return fixed_quad(f, a, b, args=args, n=1000)[0]
        #return integrate.patterson(f, a, b, epsrel = 1e-5, epsabs = 1e-10)
        #return integrate.chebyshev(f, a, b, epsrel = 1e-5, epsabs = 1e-10)
        #return integrate.romberg(f, a, b, epsrel = 1e-5, epsabs = 1e-10)
        #return integrate.quad(f, a, b, epsrel = 1e-5, epsabs = 1e-10)[0]

    cut = 1e2
    if a < cut and b > cut:
        return _int(f, a, cut) + \
                _int(lambda lz: np.exp(lz) * f(np.exp(lz)), np.log(cut), np.log(b))
    else:
        return _int(f, a, b)

@np.vectorize
def _intf(f, a, b, args=()):

    return fixed_quad(f, a, b, args=args, n=1000)[0]


if __name__=="__main__":

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import cosmology as cosmo

    c = CAMB(ini_param_file='/home/ycli/code/ksz/data/params.ini')

    z = np.array([0.1,])
    w0_list = np.array([-1.5, -1.0, -0.5])
    wa_list = np.array([-1., 0., 1.])

    pk, kh =  c.matter_power_spectrum(redshifts=z, kmin=1.e-4, kmax=2, npoints=2000)
    plt.plot(kh, pk[0], 'k-', label='1d')
    color = 'rkgb'
    for i in range(len(w0_list)):
        #c.params['w'] = w0_list[i]
        c.params['wa'] = wa_list[i]
        pk2, kh2 = c.matter_power_spectrum_2d1d(redshifts=z, kh=kh, kmax=2)
        plt.plot(kh2, pk2[0], color[i]+'--', linewidth=2., label='w=%f'%w0_list[i])

    plt.legend()
    plt.loglog()
    plt.show()

    #h = float(c.params['hubble']) / 100.
    #c2 = cosmo.Cosmology()
    #c2 = c2.init_physical(ombh2 = float(c.params['ombh2']), 
    #                      omch2 = float(c.params['omch2']), 
    #                      H0    = float(c.params['hubble']), 
    #                      omkh2 = float(c.params['omk'])*h**2)

    #z = np.linspace(0, 1, 100)

    #da = c.angular_distance(z)
    #da2= c2.angular_distance(z)

    #plt.plot(z, da, 'r-', label='camb')
    #plt.plot(z, da2,'k--', label='cosmo')
    #plt.show()






