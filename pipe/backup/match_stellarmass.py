from ksz.src import match_data

data_sets = [

    #'LOWZ_North', 
    'LOWZ_South',
    'CMASS_North',
    'CMASS_South',

    ]

for data_set in data_sets:

    catalog_path = '/data/ycli/dr12/'
    catalog = 'galaxy_DR12v5_%s_TOT_wMASS.dat'%data_set
    #stellar = 'portsmouth_stellarmass_passive_krou-DR12-boss_TOT.dat'
    stellar = 'GSWLC-A1.dat'
    #output_name = catalog_path + 'galaxy_DR12v5_%s_TOT_wMASS.dat'%data_set
    output_name = None
    #match_data.match_stellarmass_GSWLC(
    #        catalog_path, catalog, catalog_path, stellar, output_name)
    match_data.convert_mass_unit(catalog_path, catalog, h=0.6774, mode=2)




