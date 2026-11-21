import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from cement_bau.runtime import ROOT

def main():
    parser=argparse.ArgumentParser(description='Reproduce BAU and DS cement projections.')
    parser.add_argument('--scenario',default='all',type=str.lower,choices=['all','bau','baus','ds','das'])
    parser.add_argument('--workbook',type=Path,default=None)
    parser.add_argument('--country',default='all')
    parser.add_argument('--variant',default='all')
    parser.add_argument('--role',default='all',choices=['all','rep','nonrep'])
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data',help='Root containing bau/ and ds/ subdirectories')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check-inputs',action='store_true')
    parser.add_argument('--timeout',type=float,default=3600)
    args=parser.parse_args()
    scenario={'baus':'bau','das':'ds'}.get(args.scenario,args.scenario)
    manifest=json.loads((ROOT/'cement_bau/manifest.json').read_text(encoding='utf-8'))
    selected=[m for m in manifest['models'] if (scenario=='all' or m['scenario']==scenario) and (args.country=='all' or m['country']==args.country) and (args.variant=='all' or m['variant']==args.variant) and (args.role=='all' or m['scenario_role']==args.role)]
    if not selected:parser.error('No matching model; use --list. DS inputs were supplied for ten countries, excluding the US.')
    if args.list:
        for m in selected:print(f"{m['scenario']} / {m['country']} / {m['variant']} ({m['scenario_role']})")
        return 0
    for model in selected:
        model_path=ROOT/(model['module'].replace('.','/')+'.py')
        if hashlib.sha256(model_path.read_bytes()).hexdigest()!=model['packaged_sha256']:
            parser.error(f"Model differs from the verified version: {model['module']}. No models were run.")
    args.data_dir=args.data_dir.resolve();args.output_dir=args.output_dir.resolve()
    from cement_bau.workbook_io import workbook_path,validate_workbook
    if args.workbook is not None:os.environ['CEMENT_WORKBOOK']=str(args.workbook.resolve())
    consolidated=workbook_path()
    if consolidated is not None:
        try:consolidated_sha=validate_workbook(consolidated)
        except (ValueError,FileNotFoundError) as error:parser.error(str(error))
        os.environ['CEMENT_WORKBOOK']=str(consolidated)
        inputs={f"{m['scenario']}/{m['country']}":{'sha256':consolidated_sha,'source':'single consolidated workbook'} for m in selected}
        print(f'Consolidated workbook OK: {consolidated}',flush=True)
    import pandas as pd
    if consolidated is None:inputs={}
    for sc,country in sorted({(m['scenario'],m['country']) for m in selected}):
        if consolidated is not None:continue
        path=args.data_dir/sc/f'{country}.xlsx';key=f'{sc}/{country}'
        if not path.is_file():parser.error(f'Missing input: {path}')
        with pd.ExcelFile(path,engine='openpyxl') as book:
            required={m['sheet'] for m in selected if m['scenario']==sc and m['country']==country}
            if required-set(book.sheet_names):parser.error(f'Missing required worksheet in {path}')
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != manifest['inputs'][key]['sha256']:
            parser.error(f'Input differs from the original country workbook: {key}. No models were run. Restore the original input; consolidated-workbook data are not substituted into reproduction runs.')
        inputs[key]={'sha256':sha,'matches_bundled_input':sha==manifest['inputs'][key]['sha256']}
        print(f'Input OK: {key}',flush=True)
    if args.check_inputs:return 0
    versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib','openpyxl']}
    records=[]
    for model in selected:
        sc,country,variant=model['scenario'],model['country'],model['variant'];folder=args.output_dir/sc/country/variant;folder.mkdir(parents=True,exist_ok=True)
        print(f'Running {sc}/{country}/{variant}',flush=True);start=time.monotonic()
        environment=dict(os.environ,CEMENT_SCENARIO=sc,CEMENT_BAU_DATA=str(args.data_dir/sc),CEMENT_BAU_OUTPUT=str(args.output_dir/sc),MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
        record={'scenario':sc,'country':country,'variant':variant,'scenario_role':model['scenario_role'],'started_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'platform':platform.platform(),'dependencies':versions,'input':inputs[f'{sc}/{country}'],'model_sha256':hashlib.sha256((ROOT/(model['module'].replace('.','/')+'.py')).read_bytes()).hexdigest()}
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run([sys.executable,'-m','cement_bau._worker',model['module'],country,variant],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                record['returncode']=result.returncode
            except subprocess.TimeoutExpired:
                record['returncode']=124;log.write('Model timed out.\n')
        record['elapsed_seconds']=round(time.monotonic()-start,3);record['status']='passed' if record['returncode']==0 else 'failed'
        (folder/'run_metadata.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        records.append(record);print(f"{record['status']}: {sc}/{country}/{variant} ({record['elapsed_seconds']} s)",flush=True)
    (args.output_dir/'run_summary.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    return 0 if all(r['returncode']==0 for r in records) else 1

if __name__=='__main__':raise SystemExit(main())
