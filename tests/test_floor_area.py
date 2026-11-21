import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tokenize
import unittest

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'cement_bau/floor_area_manifest.json').read_text(encoding='utf-8'))


class FloorAreaTests(unittest.TestCase):
    def test_coverage_and_model_integrity(self):
        self.assertEqual(len(MANIFEST['models']),11)
        self.assertEqual(len({m['country'] for m in MANIFEST['models']}),11)
        self.assertEqual({m['scenario'] for m in MANIFEST['models']},{'bau'})
        for model in MANIFEST['models']:
            p=ROOT/(model['module'].replace('.','/')+'.py')
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),model['packaged_sha256'])

    def test_models_have_no_narratives_machine_paths_or_observed_anchor_dicts(self):
        for model in MANIFEST['models']:
            text=(ROOT/(model['module'].replace('.','/')+'.py')).read_text(encoding='utf-8')
            tree=ast.parse(text)
            self.assertFalse(any(t.type==tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(text).readline)))
            for node in ast.walk(tree):
                self.assertFalse(isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str))
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    self.assertNotRegex(node.value,r'^[A-Za-z]:[/\\]')
                    self.assertNotIn('/mnt/data',node.value)
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id.startswith('ANCHORS_'):
                            self.assertIsInstance(node.value,ast.Call)
                            self.assertEqual(ast.unparse(node.value.func),'_floor_anchors')

    def test_cli_listing_selection_and_missing_data(self):
        for args,count in [([],11),(['--country','china'],1)]:
            r=subprocess.run([sys.executable,'run_floor_area.py','--list']+args,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(len(r.stdout.splitlines()),count)
        r=subprocess.run([sys.executable,'run_floor_area.py','--country','unknown'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        r=subprocess.run([sys.executable,'run_floor_area.py','--country','china','--check-inputs','--data-dir',str(ROOT/'tests/missing_floor_data')],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
        self.assertIn('Missing private floor-area input',r.stderr)

    @unittest.skipUnless((ROOT/'data/floor_area/bau/calibration_anchors.json').is_file(),'Private data package required')
    def test_private_input_integrity(self):
        for model in MANIFEST['models']:
            for entry in model['inputs'].values():
                self.assertEqual(hashlib.sha256((ROOT/entry['file']).read_bytes()).hexdigest(),entry['sha256'])
        entry=MANIFEST['calibration_anchors']
        self.assertEqual(hashlib.sha256((ROOT/entry['file']).read_bytes()).hexdigest(),entry['sha256'])

    @unittest.skipUnless((ROOT/'data/floor_area/bau/united_states.xlsx').is_file(),'Private data package required')
    def test_changed_private_input_rejected_before_execution(self):
        original=ROOT/'data/floor_area/bau/united_states.xlsx'
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            task=Path(folder).resolve()
            self.assertTrue(task.is_relative_to(ROOT.resolve()))
            data=task/'data/floor_area/bau'
            data.mkdir(parents=True)
            (data/'united_states.xlsx').write_bytes(original.read_bytes()+b'changed-input')
            output=task/'must_not_exist'
            r=subprocess.run([sys.executable,'run_floor_area.py','--country','united_states','--data-dir',str(task/'data'),'--output-dir',str(output)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,2)
            self.assertIn('differs from the verified source',r.stderr)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
