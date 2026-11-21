import runpy
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.show = lambda *args, **kwargs: None
from cement_bau.shared_history import prepare_shared_history
prepare_shared_history(sys.argv[2], sys.argv[3])
from cement_bau.workbook_io import install_reader
install_reader(sys.argv[2],'cement')
namespace = runpy.run_module(sys.argv[1], run_name='__main__')
from cement_bau.runtime import output_dir
folder = output_dir(sys.argv[2], sys.argv[3])
for number in plt.get_fignums():
    plt.figure(number).savefig(folder / f'figure_{number:02d}.png', dpi=300, bbox_inches='tight')
plt.close('all')
