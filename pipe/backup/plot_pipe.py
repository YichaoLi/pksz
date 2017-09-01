import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ksz.src import plot

result_path = '/data/ycli/ksz/'
output_path = '/data/ycli/ksz/png/'

result_name_list = [
        'pkSZ_result_ko_tSZcln_CGC_50_AP8arcm.h5',
        'pkSZ_result_tSZcln_DR12LOWZN_1_AP8arcm.h5',
        'pkSZ_result_tSZcln_DR12CMASN_1_AP8arcm.h5',
        ]
label_list = [
        'DR7 CGC',
        'DR12 LOWZ North',
        'DR12 CMASS North',
        ]
text = '2D NIC Planck Map\nAP = $8$ arcmin'
output_name = 'plot_DR12_AP8arcm'

plot.check_TAP_bar(result_path, result_name_list,  label_list=label_list,
        output_path=output_path, output_name=output_name, text=text)
#plot.plot_cov(result_path, result_name_list, label_list=label_list,
#        output_path=output_path, text=text)
plot.plot_pkSZ(result_path, result_name_list, label_list=label_list,
        output_path=output_path, output_name=output_name,text=text)

plt.show()
