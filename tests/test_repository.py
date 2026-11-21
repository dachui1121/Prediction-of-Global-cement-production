import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'cement_bau/manifest.json').read_text(encoding='utf-8'))

class RepositoryTests(unittest.TestCase):
    def test_scenario_coverage_and_us_roles(self):
        self.assertEqual(len(MANIFEST['models']),42)
        self.assertEqual(len(MANIFEST['inputs']),21)
        for sc,n in [('bau',22),('ds',20)]:
            models=[m for m in MANIFEST['models'] if m['scenario']==sc]
            self.assertEqual(len(models),n)
            for country in {m['country'] for m in models}:
                self.assertEqual({m['scenario_role'] for m in models if m['country']==country},{'rep','nonrep'})
        us=[m for m in MANIFEST['models'] if m['country']=='united_states']
        self.assertEqual({m['scenario'] for m in us},{'bau'})
        self.assertEqual(next(m['variant'] for m in us if m['scenario_role']=='rep'),'warmup_1989')
        self.assertEqual(next(m['variant'] for m in us if m['scenario_role']=='nonrep'),'nonreplacement')

    @unittest.skipUnless((ROOT/'data/bau/united_states.xlsx').is_file(), 'Private data package required')
    def test_input_hashes(self):
        for v in MANIFEST['inputs'].values():
            self.assertEqual(hashlib.sha256((ROOT/v['file']).read_bytes()).hexdigest(),v['sha256'])

    def test_model_hashes(self):
        for v in MANIFEST['models']:
            self.assertEqual(hashlib.sha256((ROOT/(v['module'].replace('.','/')+'.py')).read_bytes()).hexdigest(),v['packaged_sha256'])

    def test_no_machine_paths_or_opening_narratives(self):
        for model in MANIFEST['models']:
            p=ROOT/(model['module'].replace('.','/')+'.py');s=p.read_text(encoding='utf-8');t=ast.parse(s)
            self.assertIsNone(ast.get_docstring(t));self.assertFalse(s.lstrip().startswith('#'))
            for n in ast.walk(t):
                if isinstance(n,ast.Constant) and isinstance(n.value,str):
                    self.assertNotRegex(n.value,r'^[A-Za-z]:[/\\]');self.assertNotIn('/mnt/data',n.value)
                if isinstance(n,ast.Call):self.assertNotIn('expanduser',ast.unparse(n.func))

    def test_cli_selection_aliases_and_errors(self):
        for args,n in [([],42),(['--scenario','BAUS'],22),(['--scenario','DAS'],20),(['--scenario','ds','--role','nonrep'],10),(['--scenario','bau','--country','united_states','--role','rep'],1)]:
            result=subprocess.run([sys.executable,'-m','cement_bau','--list']+args,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(len(result.stdout.splitlines()),n)
        result=subprocess.run([sys.executable,'-m','cement_bau','--scenario','ds','--country','united_states'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)

    def test_missing_inputs(self):
        result=subprocess.run([sys.executable,'-m','cement_bau','--scenario','ds','--country','china','--check-inputs','--data-dir',str(ROOT/'tests/nonexistent_data')],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('Missing input',result.stderr)

    def test_live_history_code_does_not_read_reference_outputs(self):
        for name in ['shared_history.py','_history_worker.py']:
            s=(ROOT/'cement_bau'/name).read_text(encoding='utf-8')
            self.assertNotIn('reference_results',s)
            self.assertNotIn('read_excel',s)
        config=json.loads((ROOT/'cement_bau/history_config.json').read_text())
        self.assertEqual(set(config),{'bau','ds'})

    def test_unit_mapping(self):
        for m in MANIFEST['models']:
            self.assertEqual(m['output_to_gt'],0.1 if m['country'] in ['russia','united_kingdom','iran'] else 1e-9)

    @unittest.skipUnless((ROOT/'data/bau/united_states.xlsx').is_file(), 'Private data package required')
    def test_modified_input_is_rejected_before_model_execution(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            task_path=Path(folder).resolve()
            self.assertTrue(task_path.is_relative_to(ROOT.resolve()))
            data=task_path/'data/bau';data.mkdir(parents=True)
            original=ROOT/'data/bau/united_states.xlsx'
            (data/'united_states.xlsx').write_bytes(original.read_bytes()+b'changed input fixture')
            output=task_path/'must_not_be_created'
            result=subprocess.run([sys.executable,'-m','cement_bau','--scenario','bau','--country','united_states','--data-dir',str(task_path/'data'),'--output-dir',str(output)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Input differs from the original country workbook',result.stderr)
            self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
