import hashlib
import json
import os
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=json.loads((ROOT/'cement_bau/workbook_schema.json').read_text(encoding='utf-8'))
_FRAMES={}
_BOOK=None
_ORIGINAL_READ=pd.read_excel
_ORIGINAL_EXCEL=pd.ExcelFile


def workbook_path():
    explicit=os.environ.get('CEMENT_WORKBOOK')
    if explicit:return Path(explicit).resolve()
    default=ROOT/'data/Research_Data.xlsx'
    return default.resolve() if default.is_file() else None


def validate_workbook(path):
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(f'Missing consolidated input workbook: {path}')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=SCHEMA['workbook_sha256']:
        raise ValueError('Consolidated workbook differs from the verified data release. Use the matching code and workbook versions.')
    return digest


def frame_for(file,sheet):
    global _BOOK
    entries=SCHEMA['datasets'][file]
    if isinstance(sheet,int):entry=entries[sheet]
    else:entry=next(e for e in entries if e['legacy_sheet']==sheet)
    key=entry['sheet']
    if key not in _FRAMES:
        if _BOOK is None:_BOOK=openpyxl.load_workbook(workbook_path(),read_only=True,data_only=True)
        ws=_BOOK[key]
        rows=list(ws.iter_rows(min_row=1,max_row=entry['rows']+1,max_col=entry['cols'],values_only=True))
        if list(rows[0])!=entry['english_columns']:raise ValueError(f'Changed input column layout: {key}')
        restored=[]
        for row in rows[1:]:
            restored.append([SCHEMA['text_translations'].get(v,v) if isinstance(v,str) else v for v in row])
        frame=pd.DataFrame(restored,columns=entry['columns'])
        for col,dtype in zip(entry['columns'],entry['dtypes']):
            if dtype=='object':
                frame[col]=frame[col].astype(object).where(pd.notna(frame[col]),np.nan)
            else:frame[col]=frame[col].astype(dtype)
        _FRAMES[key]=frame
    return _FRAMES[key].copy(deep=True)


def current_file(country,kind):
    if kind=='cement':return f"data/{os.environ.get('CEMENT_SCENARIO','bau')}/{country}.xlsx"
    manifest=json.loads((ROOT/'cement_bau/floor_area_manifest.json').read_text(encoding='utf-8'))
    return next(m for m in manifest['models'] if m['country']==country)['inputs']['macro']['file']


def install_reader(country,kind):
    path=workbook_path()
    if path is None:return
    file=current_file(country,kind)
    sheets=[e['legacy_sheet'] for e in SCHEMA['datasets'][file]]

    def matches(value):
        if isinstance(value,ConsolidatedExcel):return True
        try:return Path(value).resolve()==path
        except TypeError:return False

    class ConsolidatedExcel:
        def __init__(self,*args,**kwargs):self.sheet_names=list(sheets)
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def close(self):return None
        def parse(self,sheet_name=0,**kwargs):return read(path,sheet_name=sheet_name,**kwargs)

    def excel(value,*args,**kwargs):
        return ConsolidatedExcel() if matches(value) else _ORIGINAL_EXCEL(value,*args,**kwargs)

    def read(value,sheet_name=0,*args,**kwargs):
        if not matches(value):return _ORIGINAL_READ(value,sheet_name=sheet_name,*args,**kwargs)
        if sheet_name is None:return {s:read(value,sheet_name=s,**kwargs) for s in sheets}
        if isinstance(sheet_name,list):return {s:read(value,sheet_name=s,**kwargs) for s in sheet_name}
        result=frame_for(file,sheet_name)
        usecols=kwargs.get('usecols')
        if isinstance(usecols,str):
            columns=[]
            from openpyxl.utils.cell import column_index_from_string
            for part in usecols.split(','):
                bounds=part.strip().split(':')
                lo=column_index_from_string(bounds[0])-1
                hi=column_index_from_string(bounds[-1])
                columns.extend(range(lo,hi))
            result=result.iloc[:,columns]
        elif usecols is not None:
            result=result.iloc[:,usecols] if all(isinstance(v,int) for v in usecols) else result.loc[:,usecols]
        if kwargs.get('nrows') is not None:result=result.iloc[:kwargs['nrows']]
        return result

    pd.read_excel=read
    pd.ExcelFile=excel


def consolidated_anchors(country):
    global _BOOK
    if _BOOK is None:_BOOK=openpyxl.load_workbook(workbook_path(),read_only=True,data_only=True)
    return {int(row[1]):row[2] for row in _BOOK[SCHEMA['anchors_sheet']].iter_rows(min_row=2,values_only=True) if row[0]==country}
