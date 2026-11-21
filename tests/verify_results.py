import argparse
import json
from pathlib import Path
import numpy as np
import openpyxl

ROOT=Path(__file__).resolve().parents[1]

def read_series(path,model):
    book=openpyxl.load_workbook(path,read_only=True,data_only=True)
    try:
        rows=iter(book[model['result_sheet']].iter_rows(values_only=True));headers=list(next(rows));yi=headers.index(model['year_column']);vi=headers.index(model['value_column'])
        return {int(r[yi]):r[vi] for r in rows if isinstance(r[yi],(int,float))}
    finally:book.close()

def verify(results,baseline,scenario='all',write_report=True):
    manifest=json.loads((ROOT/'cement_bau/manifest.json').read_text(encoding='utf-8'))
    models=[m for m in manifest['models'] if scenario=='all' or m['scenario']==scenario]
    reports=[]
    for m in models:
        sc,country,variant=m['scenario'],m['country'],m['variant'];folder=results/sc/country/variant
        files=list(folder.glob('*.xlsx'));assert len(files)==1,(sc,country,variant,'missing/duplicate workbook')
        current=read_series(files[0],m);original=read_series(baseline/sc/country/f'{variant}.xlsx',m)
        future=[y for y in range(2025,2101) if current.get(y)!=original.get(y) or current.get(y) is None]
        assert not future,(sc,country,variant,'forecast mismatch',future)
        record={'scenario':sc,'country':country,'variant':variant,'role':m['scenario_role'],'forecast_exact_matches':76}
        if m['scenario_role']=='nonrep':
            rep=next(x for x in models if x['scenario']==sc and x['country']==country and x['scenario_role']=='rep')
            upper=read_series(baseline/sc/country/f"{rep['variant']}.xlsx",rep)
            mismatch=[y for y in range(1990,2025) if current.get(y)!=upper.get(y) or current.get(y) is None]
            assert not mismatch,(sc,country,'history mismatch',mismatch)
            with np.load(folder/'shared_history_state.npz',allow_pickle=False) as state:
                mask=(state['years']>=2025)&(state['years']<=2100)
                assert mask.sum()==76
                assert state['forecast_original'][mask].tobytes()==state['combined'][mask].tobytes(),(sc,country,'raw forecast bytes changed')
                hm=(state['years']>=1990)&(state['years']<=2024)
                source={int(y):v for y,v in zip(state['rep_years'],state['rep_values'])}
                assert all(v==source[int(y)] for y,v in zip(state['years'][hm],state['combined'][hm]))
            record['history_exact_matches']=35;record['raw_forecast_bytes_unchanged']=True
        reports.append(record)
        print(f'PASS {sc}/{country}/{variant}: forecast exact'+('; shared history exact' if m['scenario_role']=='nonrep' else ''))
    if write_report:(results/'verification.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    return reports

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'results');p.add_argument('--baseline',type=Path,default=ROOT/'reference_results');p.add_argument('--scenario',default='all',choices=['all','bau','ds']);a=p.parse_args();verify(a.results,a.baseline,a.scenario)

if __name__=='__main__':main()
