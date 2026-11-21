from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

def input_path(country):
    from cement_bau.workbook_io import workbook_path
    consolidated=workbook_path()
    if consolidated is not None:return consolidated
    path = Path(os.environ.get('CEMENT_BAU_DATA', ROOT / 'data' / os.environ.get('CEMENT_SCENARIO', 'bau'))).resolve() / f'{country}.xlsx'
    if not path.is_file():
        raise FileNotFoundError(f'Missing input workbook: {path}')
    return path

def output_dir(country, variant):
    path = Path(os.environ.get('CEMENT_BAU_OUTPUT', ROOT / 'results' / os.environ.get('CEMENT_SCENARIO', 'bau'))).resolve() / country / variant
    path.mkdir(parents=True, exist_ok=True)
    return path
