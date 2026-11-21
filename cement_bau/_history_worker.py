import os
import json
import runpy
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cement_bau.runtime import ROOT

plt.show = lambda *args, **kwargs: None
country, destination = sys.argv[1:3]
config = json.loads((ROOT / 'cement_bau/history_config.json').read_text(encoding='utf-8'))[os.environ.get('CEMENT_SCENARIO', 'bau')][country]
from cement_bau.workbook_io import install_reader
install_reader(country,'cement')
namespace = runpy.run_module(config['rep_module'], run_name='__main__')
np.savez(destination, years=np.asarray(namespace[config['rep_years']]), values=np.asarray(namespace[config['rep_series']], dtype=float))
plt.close('all')
