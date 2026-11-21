import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from cement_bau.runtime import ROOT
from cement_bau.floor_area_runtime import MANIFEST


def main():
    parser = argparse.ArgumentParser(description='BAU residential floor-area fits for 11 countries.')
    parser.add_argument('--workbook',type=Path,default=None)
    parser.add_argument('--country', default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'floor_area_results')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    selected = [m for m in MANIFEST['models'] if args.country in ['all',m['country']]]
    if not selected:parser.error('No matching floor-area model; use --list.')
    if args.list:
        for m in selected:print(f"bau / {m['country']} / floor_area")
        return 0
    data_root = args.data_dir.resolve()
    inputs = {}
    from cement_bau.workbook_io import workbook_path,validate_workbook
    if args.workbook is not None:os.environ['CEMENT_WORKBOOK']=str(args.workbook.resolve())
    consolidated=workbook_path()
    if consolidated is not None:
        try:inputs['consolidated']=validate_workbook(consolidated)
        except (ValueError,FileNotFoundError) as error:parser.error(str(error))
        os.environ['CEMENT_WORKBOOK']=str(consolidated)
    for m in selected:
        path = ROOT/(m['module'].replace('.','/')+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=m['packaged_sha256']:
            parser.error(f"Model differs from the verified version: {m['module']}")
        if consolidated is not None:continue
        required = list(m['inputs'].values())
        if m['uses_private_anchors']:required.append(MANIFEST['calibration_anchors'])
        for entry in required:
            key = entry['file']
            path = data_root/Path(key).relative_to('data')
            if not path.is_file():parser.error(f'Missing private floor-area input: {path}')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=entry['sha256']:parser.error(f'Floor-area input differs from the verified source: {key}')
            inputs[key] = digest
    print(f'Input checks passed: {len(selected)} floor-area models',flush=True)
    if args.check_inputs:return 0
    output = args.output_dir.resolve()
    records = []
    dependencies = {k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','matplotlib','openpyxl']}
    for m in selected:
        folder = output/'bau'/m['country']
        folder.mkdir(parents=True,exist_ok=True)
        print(f"Running floor_area/bau/{m['country']}",flush=True)
        start = time.monotonic()
        environment = dict(os.environ,CEMENT_FLOOR_DATA=str(data_root),CEMENT_FLOOR_OUTPUT=str(output),MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            try:
                result = subprocess.run([sys.executable,'-m','cement_bau._floor_area_worker',m['module'],m['country']],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write('Model timed out.\n')
        record = {'country':m['country'],'scenario':'bau','returncode':code,'elapsed_seconds':round(time.monotonic()-start,3),
                  'python':platform.python_version(),'dependencies':dependencies,'model_sha256':m['packaged_sha256'],'input_sha256':inputs}
        (folder/'run_metadata.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        records.append(record)
        print(f"{'passed' if code==0 else 'failed'}: floor_area/bau/{m['country']} ({record['elapsed_seconds']} s)",flush=True)
    (output/'run_summary.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    return 0 if all(r['returncode']==0 for r in records) else 1
