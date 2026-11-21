import json
import os
from pathlib import Path
from cement_bau.runtime import ROOT

MANIFEST = json.loads((ROOT/'cement_bau/floor_area_manifest.json').read_text(encoding='utf-8'))


def floor_input(country, role='macro'):
    from cement_bau.workbook_io import workbook_path
    consolidated=workbook_path()
    if consolidated is not None:return consolidated
    model = next(m for m in MANIFEST['models'] if m['country']==country)
    root = Path(os.environ.get('CEMENT_FLOOR_DATA', ROOT/'data')).resolve()
    return root/Path(model['inputs'][role]['file']).relative_to('data')


def floor_output(country):
    root = Path(os.environ.get('CEMENT_FLOOR_OUTPUT', ROOT/'floor_area_results')).resolve()
    folder = root/'bau'/country
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def floor_anchors(country):
    from cement_bau.workbook_io import workbook_path,consolidated_anchors
    if workbook_path() is not None:return consolidated_anchors(country)
    root = Path(os.environ.get('CEMENT_FLOOR_DATA', ROOT/'data')).resolve()
    path = root/Path(MANIFEST['calibration_anchors']['file']).relative_to('data')
    data = json.loads(path.read_text(encoding='utf-8'))[country]
    return {int(year): value for year,value in data.items()}
