from ksz.src import models

output_path = '/home/ycli/code/ksz/data/'

# 6dF 
logmass_min = 11.
redshift = 0.05
output_name = 'v_logmassmin%d_z%3.2f.h5'%(logmass_min, redshift)
models.velocity_mnu(output_path, output_name, logmass_min, redshift)

# dr7 cgc
logmass_min = 11.
redshift = 0.1
output_name = 'v_logmassmin%d_z%3.2f.h5'%(logmass_min, redshift)
models.velocity_mnu(output_path, output_name, logmass_min, redshift)

# dr12 lowz
logmass_min = 13.
redshift = 0.2
output_name = 'v_logmassmin%d_z%3.2f.h5'%(logmass_min, redshift)
models.velocity_mnu(output_path, output_name, logmass_min, redshift)

# dr12 cmass
logmass_min = 13.
redshift = 0.5
output_name = 'v_logmassmin%d_z%3.2f.h5'%(logmass_min, redshift)
models.velocity_mnu(output_path, output_name, logmass_min, redshift)

