import json
import runpy
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cement_bau.floor_area_runtime import floor_output

plt.show = lambda *a, **k: None
from cement_bau.workbook_io import install_reader
install_reader(sys.argv[2],'floor')
namespace = runpy.run_module(sys.argv[1],run_name='__main__')
frame = namespace['result_df'] if 'result_df' in namespace else namespace['out']
areas = [c for c in frame.columns if '面积' in c and '预测' in c]
assert len(areas)==1, areas
folder = floor_output(sys.argv[2])
np.savez(folder/'series_state.npz',years=frame['年份'].to_numpy(),area=frame[areas[0]].to_numpy(),columns=np.array(list(frame.columns),dtype=str),all_values=frame.to_numpy(dtype=float))
(folder/'series_metadata.json').write_text(json.dumps({'year_column':'年份','area_column':areas[0],'period':[int(frame['年份'].min()),int(frame['年份'].max())]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for number in plt.get_fignums():
    plt.figure(number).savefig(folder/f'figure_{number:02d}.png',dpi=200,bbox_inches='tight')
plt.close('all')
