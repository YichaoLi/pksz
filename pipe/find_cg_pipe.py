
from ksz.src import find_cg


#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_CMASS_South_RaDecZ.dat'
#find_cg.find_GC(input_path, input_name, criterion=1., bcut=50.)
#    
#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_CMASS_North_RaDecZ.dat'
#find_cg.find_GC(input_path, input_name, criterion=1., bcut=50.)
#
#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_LOWZ_South_RaDecZ.dat'
#find_cg.find_GC(input_path, input_name, criterion=1., bcut=50.)
#    
#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_LOWZ_North_RaDecZ.dat'
#find_cg.find_GC(input_path, input_name, criterion=1., bcut=50.)

#input_path = '/data/ycli/6df/'
##input_name = '6df_ra_dec_z_6dFGSzDR3.cat'
##input_name = '6df_ra_dec_z_k1275_wufq_totcompl_K70_2.cat'
#input_name = '6dFGS_2MASS_RA_DEC_Z_J_K_bJ_rF_GOOD.cat'
#find_cg.find_GC(input_path, input_name, criterion=1., bcut=100.)

#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_LOWZ_North_TOT_wMASS.dat'
#find_cg.find_GC(input_path, input_name, criterion=[1, 2, 4])

#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_LOWZ_South_TOT_wMASS.dat'
#find_cg.find_GC(input_path, input_name, criterion=[1, 2, 4])
#
#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_CMASS_North_TOT_wMASS.dat'
#find_cg.find_GC(input_path, input_name, criterion=[1, 2, 4])
#
#input_path = '/data/ycli/dr12/'
#input_name = 'galaxy_DR12v5_CMASS_South_TOT_wMASS.dat'
#find_cg.find_GC(input_path, input_name, criterion=[1, 2, 4])

input_path = '/data/ycli/group_catalog/'
input_name = 'SDSS_M_group.dat'
find_cg.find_GC(input_path, input_name, criterion=[1, 2, 4])

