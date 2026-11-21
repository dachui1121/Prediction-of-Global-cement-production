import argparse
import json
from pathlib import Path
import numpy as np
import openpyxl

ROOT=Path(__file__).resolve().parents[1]


def series(path,model):
    book=openpyxl.load_workbook(path,read_only=True,data_only=True)
    try:
        rows=iter(book[model['result_sheet']].values)
        headers=list(next(rows))
        yi,vi=headers.index(model['year_column']),headers.index(model['value_column'])
        return {int(r[yi]):r[vi]*model['output_to_gt'] for r in rows if isinstance(r[yi],(int,float)) and 1990<=r[yi]<=2100}
    finally:book.close()


def verify(workbook,cement_results=None,floor_results=None):
    source=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
    report={}
    if cement_results is not None:
        reference={}
        for row in source['Cement Reference'].iter_rows(min_row=2,values_only=True):
            reference.setdefault((row[0],row[1],row[2]),{})[int(row[4])]=row[5]
        manifest=json.loads((ROOT/'cement_bau/manifest.json').read_text(encoding='utf-8'))
        points=0
        for m in manifest['models']:
            folder=cement_results/m['scenario']/m['country']/m['variant']
            files=list(folder.glob('*.xlsx'))
            assert len(files)==1,(folder,files)
            current=series(files[0],m)
            expected=reference[(m['scenario'],m['country'],m['scenario_role'])]
            assert set(current)==set(expected)==set(range(1990,2101))
            assert all(abs(current[y]-expected[y])<=1e-12 for y in expected),(m['scenario'],m['country'],m['variant'])
            if m['scenario_role']=='nonrep':
                with np.load(folder/'shared_history_state.npz',allow_pickle=False) as state:
                    mask=(state['years']>=2025)&(state['years']<=2100)
                    assert state['forecast_original'][mask].tobytes()==state['combined'][mask].tobytes()
            points+=111
            print(f"PASS cement/{m['scenario']}/{m['country']}/{m['variant']}")
        report['cement_annual_values']=points
    if floor_results is not None:
        reference={}
        for row in source['Floor Area Reference'].iter_rows(min_row=2,values_only=True):
            reference.setdefault(row[0],{})[int(row[1])]=row[2]
        manifest=json.loads((ROOT/'cement_bau/floor_area_manifest.json').read_text(encoding='utf-8'))
        points=0
        for m in manifest['models']:
            with np.load(floor_results/'bau'/m['country']/'series_state.npz',allow_pickle=False) as state:
                current={int(y):v for y,v in zip(state['years'],state['area'])}
            expected=reference[m['country']]
            assert set(current)==set(expected)
            assert all(abs(current[y]-expected[y])<=1e-10 for y in expected),m['country']
            points+=len(current)
            print(f"PASS floor_area/bau/{m['country']}")
        report['floor_area_annual_values']=points
    source.close()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workbook',type=Path,default=ROOT/'data/Research_Data.xlsx')
    p.add_argument('--cement-results',type=Path)
    p.add_argument('--floor-results',type=Path)
    args=p.parse_args()
    if args.cement_results is None and args.floor_results is None:p.error('Provide --cement-results and/or --floor-results.')
    print(json.dumps(verify(args.workbook,args.cement_results,args.floor_results),indent=2))
