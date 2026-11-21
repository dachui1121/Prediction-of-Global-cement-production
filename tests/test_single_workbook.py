import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SingleWorkbookTests(unittest.TestCase):
    def test_schema_covers_all_model_inputs(self):
        schema=json.loads((ROOT/'cement_bau/workbook_schema.json').read_text(encoding='utf-8'))
        cement=json.loads((ROOT/'cement_bau/manifest.json').read_text(encoding='utf-8'))
        floor=json.loads((ROOT/'cement_bau/floor_area_manifest.json').read_text(encoding='utf-8'))
        for entry in cement['inputs'].values():self.assertIn(entry['file'],schema['datasets'])
        for model in floor['models']:
            for entry in model['inputs'].values():self.assertIn(entry['file'],schema['datasets'])
        self.assertEqual(len(schema['datasets']),26)
        self.assertEqual(sum(len(x) for x in schema['datasets'].values()),27)
        self.assertEqual(len(schema['workbook_sha256']),64)

    def test_missing_or_modified_workbook_rejected_before_model_execution(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            folder=Path(directory)
            modified=folder/'bad.xlsx'
            modified.write_bytes(b'not the verified workbook')
            for runner in ['run.py','run_floor_area.py']:
                for path,message in [(folder/'missing.xlsx','Missing consolidated input workbook'),(modified,'differs from the verified data release')]:
                    output=folder/'must_not_exist'
                    result=subprocess.run([sys.executable,runner,'--country','china','--workbook',str(path),'--output-dir',str(output)],cwd=ROOT,capture_output=True,text=True)
                    self.assertEqual(result.returncode,2,result.stderr)
                    self.assertIn(message,result.stderr)
                    self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
