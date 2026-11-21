import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def verify(results,baseline):
    manifest=json.loads((ROOT/'cement_bau/floor_area_manifest.json').read_text(encoding='utf-8'))
    reports=[]
    for model in manifest['models']:
        relative=Path('bau')/model['country']/'series_state.npz'
        with np.load(results/relative,allow_pickle=False) as current,np.load(baseline/relative,allow_pickle=False) as original:
            for field in ['years','columns']:
                assert np.array_equal(current[field],original[field]),(model['country'],field)
            for field in ['area','all_values']:
                assert np.array_equal(current[field],original[field],equal_nan=True),(model['country'],field)
            reports.append({'country':model['country'],'annual_values':len(current['years']),'all_numeric_columns_exact':True})
            print(f"PASS floor_area/bau/{model['country']}: all numeric columns exact")
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'floor_area_results')
    parser.add_argument('--baseline',type=Path,default=ROOT/'floor_area_reference')
    args=parser.parse_args()
    verify(args.results,args.baseline)
