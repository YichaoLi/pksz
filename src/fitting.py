import numpy as np


def searching_limit(chisq):

    L = np.exp(-0.5 * chisq)

    L_argmax = L.argmax()
    L_step = L[1] - L[0]
    # searching upper limit
    p_upper_arg = L_argmax
    p_upper_total = L[L_argmax:-1].sum() * L_step - L[L_argmax] * 0.5 * L_step
    p_upper = L[L_argmax] * 0.5 * L_step
    for i in range(L_argmax + 1, len(L)):
        p_upper += L[i] * L_step
        if p_upper/p_upper_total > 0.68:
            p_upper_arg = i
            break
    # searching lower limit
    p_lower_arg = L_argmax
    p_lower_total = L[0:L_argmax+1].sum() * L_step - L[L_argmax] * 0.5 * L_step
    p_lower = L[L_argmax] * 0.5 * L_step
    for i in range(L_argmax - 1, 0, -1):
        p_lower += L[i] * L_step
        if p_lower/p_lower_total > 0.68:
            p_lower_arg = i
            break
    return p_upper_arg, p_lower_arg

def amp_fitting(pksz_obs, pksz_th, pksz_err=None, pksz_covi=None, T_CMB=2.7255):

    tau_bar = np.linspace(-1.0, 3.0, 500) * 1.e-4

    factor = T_CMB * 1.e6 / 2.99e5

    amp = tau_bar[:, None] * factor

    if pksz_covi is not None:
        #pksz_covi = np.linalg.inv(pksz_cov)
        #x = pksz_th[None, :] * amp - pksz_obs[None, :]
        #chisq = np.sum(np.tensordot(x, pksz_covi, axes=(1, 0)) * x, axis=1)
        nor = np.dot(np.dot(pksz_th, pksz_covi), pksz_th)
        amp = np.dot(np.dot(pksz_obs, pksz_covi), pksz_th) / nor
        sig = np.sqrt(1. / nor)

        x = pksz_obs - pksz_th * amp
        chisq = np.dot(np.dot(x, pksz_covi), x)
        chisq /= float(x.shape[0] - 1)

        #pksz_covi = np.linalg.inv(pksz_cov[sel,:][:,sel])
        chisq_null = np.dot(np.dot(pksz_obs, pksz_covi), pksz_obs)
        chisq_null /= float(pksz_obs.shape[0] - 1)

        return amp, amp + sig, amp - sig, factor, chisq_null, chisq

    elif pksz_err is not None:
        chisq  = (pksz_th[None, :] * amp[:, None] - pksz_obs[None, :])**2.
        chisq /= pksz_err[None, :] ** 2.
        chisq  = np.sum(chisq, axis=-1)
        chisq -= np.min(chisq)

        chisq_null = np.sum(pksz_obs / pksz_err)

        chisq_min = np.argmin(chisq)

        upper, lower = searching_limit(chisq)

        #val_best = amp[chisq_min]/factor*1.e4
        #sig_plus = amp[upper]/factor*1.e4 - amp[chisq_min]/factor*1.e4
        #sig_mins = amp[chisq_min]/factor*1.e4 - amp[lower]/factor*1.e4

        #print '-'*20
        #print "tau = %4.2f +%4.2f/-%4.2f (+%4.2f/-%4.2f)sigma"%(
        #        val_best, sig_plus, sig_mins, val_best/sig_plus, val_best/sig_mins)
        #print

        return amp[chisq_min][0], amp[upper][0], amp[lower][0], factor, chisq_null, \
                chisq[chisq_min] / float(pksz_obs.shape[0] - 1)

