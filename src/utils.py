import time
import numpy as np
import logging as log
import ephem as e

@np.vectorize
def convert_eq2ga(ra, dec, i_epoch=e.J2000, o_epoch=None, degree_in=True, 
        degree_out=True):

    if o_epoch is None:
        o_epoch = i_epoch

    if degree_in:
        ra  = e.hours(ra*np.pi/180.)
        dec = e.degrees(dec*np.pi/180.)
    else:
        ra  = e.hours(ra*np.pi)
        dec = e.degrees(dec*np.pi)
    
    coord = e.Equatorial(ra, dec, epoch=i_epoch)
    coord = e.Galactic(coord, epoch=o_epoch)

    l, b = coord.lon, coord.lat

    if degree_out:
        l = l * 180./np.pi
        b = b * 180./np.pi

    return l, b



def log_timing_func():
    '''Decorator generator that logs the time it takes a function to execute'''
    def decorator(func_to_decorate):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func_to_decorate(*args, **kwargs)
            elapsed = (time.time() - start)

            log.debug("[TIMING] %s: %s" % (func_to_decorate.__name__, elapsed))

            return result
        wrapper.__doc__ = func_to_decorate.__doc__
        wrapper.__name__ = func_to_decorate.__name__
        return wrapper
    return decorator


# for classes
def log_timing(func_to_decorate):
    '''Decorator generator that logs the time it takes a function to execute'''
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func_to_decorate(*args, **kwargs)
        elapsed = (time.time() - start)

        #log.debug("[TIMING] %s: %s" % (func_to_decorate.__name__, elapsed))
        print "[TIMING] %s: %s" % (func_to_decorate.__name__, elapsed)

        return result

    wrapper.__doc__ = func_to_decorate.__doc__
    wrapper.__name__ = func_to_decorate.__name__
    return wrapper


def gen_loop_N_old(mat_len):

    max_len = int(mat_len**0.5)
    loop_N   = (mat_len / max_len) + 1
    len_list = np.arange(loop_N)
    len_list = max_len * (1./(len_list+1.))
    len_list = len_list[len_list>1]
    len_list = len_list.astype('int')
    len_list = np.append(len_list, np.ones(mat_len-len_list.sum()).astype('int'))
    loop_N   = len(len_list)

    return loop_N, len_list

def gen_loop_N(mat_len, l_max):

    l_min = l_max**2/mat_len
    if l_min < 1: l_min = 1
    if l_min > l_max: l_min = l_max

    #l_max = int(mat_len**0.5)
    loop_N   = (mat_len / l_max) + 1
    len_list = np.arange(loop_N)
    len_list = l_max * (1./(len_list+1.))
    len_list = len_list[len_list>l_min]
    len_list = len_list.astype('int')
    loop_N_ext = int((mat_len-len_list.sum())/l_min) + 1
    len_list = np.append(len_list, (np.ones(loop_N_ext)*l_min).astype('int'))
    loop_N   = len(len_list)

    return loop_N, len_list

def gen_loop_N_better(mat_len, l_max):

    N = l_max**2

    len_list = [l_max, ]
    while sum(len_list) < mat_len:
        l_x = len_list[-1]
        l_y = sum(len_list)

        l_x = int(0.5 * ( np.sqrt(l_y**2 + 4. * N) - l_y ))

        len_list.append(l_x)

    len_list = np.array(len_list)
    loop_N   = len(len_list)

    return loop_N, len_list

def gen_jk_sample(jk_n, jk_len, cij, index_st):

    jk_st = jk_len * jk_n
    jk_ed = jk_len * (jk_n + 1)

    cij[jk_st : jk_ed, :] = 0
    cij[:, jk_st - index_st : jk_ed - index_st] = 0

    return cij

if __name__=="__main__":

    mat_len = 260000
    loop_N, len_list = gen_loop_N_better(mat_len, l_max=9000)
    print '-'*80
    print loop_N


    #loop_N, len_list = gen_loop_N(mat_len, l_max=5000)
    #print '-'*80
    #print loop_N
    #for i in range(loop_N):
    #    print len_list[i], len_list[i]*len_list[:i+1].sum(), len_list[:i].sum(), len_list[:i+1].sum()

