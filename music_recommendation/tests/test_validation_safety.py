"""Synthetic regressions only: no dataset, training or final scoring."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_windows


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class AcceptanceSafetyTests(unittest.TestCase):
    def run_python(self, code, optimized=True):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), str(ROOT/'scripts')])
        return subprocess.run([sys.executable] + (['-O'] if optimized else []) + ['-c', code],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def test_duplicate_matrix_rejected_with_and_without_optimization(self):
        code = """import pandas as pd
from musicrec.data import indexed_matrix
indexed_matrix(pd.DataFrame({'userID':[1,1], 'artistID':[2,2]}), pd.DataFrame({'id':[2]}))
"""
        for optimized in (False, True):
            process = self.run_python(code, optimized)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('Consistency check failed', process.stderr)

    def test_bad_final_manifest_rejected_under_optimization(self):
        process = self.run_python("""import json, tempfile
from pathlib import Path
import summarize_final as s
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary); out=root/'outputs/final'; out.mkdir(parents=True)
    (out/'evaluation_state.json').write_text(json.dumps({'status':'completed','evaluated_runs':11}))
    (out/'final_checkpoint_manifest.json').write_text('[]')
    s.verify_freeze=lambda root: ({}, {})
    s.verify_and_summarize(root)
""")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Consistency check failed', process.stderr)

    def test_runtime_sources_have_no_asserts(self):
        for directory in ('src', 'scripts'):
            for path in (ROOT/directory).rglob('*.py'):
                tree=ast.parse(path.read_text())
                self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)), str(path))

    def test_template_test_failures_propagate_under_optimization(self):
        import nbformat
        for generator in verify_windows.NOTEBOOKS.values():
            spec=importlib.util.spec_from_file_location('template', ROOT/'scripts'/generator)
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as temporary:
                module.ROOT=Path(temporary); (module.ROOT/'notebooks').mkdir()
                module.main()
                notebook=nbformat.read(next((module.ROOT/'notebooks').glob('*.ipynb')),as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type=='code':
                        self.assertFalse(any(isinstance(n,ast.Assert) for n in ast.walk(ast.parse(cell.source))))
                failure_cell=next(c.source for c in notebook.cells if c.cell_type=='code' and 'check_returncode' in c.source)
                code="""import subprocess, sys
from pathlib import Path
ROOT=Path.cwd()
subprocess.run=lambda *a,**kw: subprocess.CompletedProcess(['synthetic-failure'], 7, '', 'fixture failure')
"""+failure_cell
                process=self.run_python(code)
                self.assertNotEqual(process.returncode,0)
                self.assertIn('CalledProcessError',process.stderr)

    def test_each_notebook_is_isolated_and_source_unchanged(self):
        import nbformat
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary); source=base/'source'; output=base/'results'; output.mkdir()
            for dirname in ('src','scripts','tests','data','outputs','notebooks'):
                (source/dirname).mkdir(parents=True)
            (source/'outputs/frozen.txt').write_text('original evidence')
            (source/'data/synthetic.csv').write_text('synthetic only')
            (source/'notebooks/original.ipynb').write_text('original notebook evidence')
            for dirname in ('.venv','__pycache__','deliverables','windows_native_verification'):
                (source/'outputs'/dirname).mkdir()
                (source/'outputs'/dirname/'excluded').write_text('must not copy')
            for filename,generator in verify_windows.NOTEBOOKS.items():
                (source/'scripts'/generator).write_text(
                    'import nbformat\nfrom pathlib import Path\n'
                    f"nbformat.write(nbformat.v4.new_notebook(),Path('notebooks')/{filename!r})\n")
            before=snapshot(source); workspaces=[]
            class SyntheticClient:
                def __init__(self, notebook, **kwargs):
                    self.notebook=notebook
                    self.work=Path(kwargs['resources']['metadata']['path'])
                def execute(self, **kwargs):
                    workspaces.append(self.work)
                    self_outer.assertNotEqual(self.work,source)
                    self_outer.assertEqual((self.work/'outputs/frozen.txt').read_text(),'original evidence')
                    for name in verify_windows.EXCLUDED:
                        self_outer.assertFalse((self.work/'outputs'/name).exists())
                    # Exercise the injected interpreter guard, including optimized mode.
                    exec(compile(self.notebook.cells[0].source,'kernel-check','exec',optimize=2),{})
                    with patch.object(sys, 'executable', str(self.work/'wrong-python')):
                        with self_outer.assertRaisesRegex(RuntimeError, 'Wrong Python kernel'):
                            exec(compile(self.notebook.cells[0].source,'kernel-check','exec',optimize=2),{})
                    (self.work/'outputs/frozen.txt').write_text('overwritten in sandbox')
                    (self.work/'data/synthetic.csv').write_text('overwritten in sandbox')
            self_outer=self
            for filename in verify_windows.NOTEBOOKS:
                verify_windows.execute_notebook(source,output,filename,os.environ.copy(),SyntheticClient)
            self.assertEqual(snapshot(source),before)
            self.assertEqual(len(set(workspaces)),3)
            self.assertTrue(all(not p.exists() for p in workspaces))
            self.assertEqual(len(list(output.glob('*.ipynb'))),3)

    def test_notebook_failure_propagates_and_cleans_workspace(self):
        # A missing generator simulates a subprocess failure before kernel launch.
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'scripts').mkdir()
            with self.assertRaises(subprocess.CalledProcessError):
                verify_windows.execute_notebook(root,root,next(iter(verify_windows.NOTEBOOKS)),os.environ.copy())

    def test_runner_records_failure_instead_of_pass(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            failure=subprocess.CompletedProcess(['synthetic-tests'],3,'','fixture failure')
            with patch.object(verify_windows,'ROOT',root), patch.object(sys,'argv',['verify_windows']), \
                 patch.object(verify_windows.platform,'system',return_value='Windows'), \
                 patch.object(verify_windows.subprocess,'run',return_value=failure):
                with self.assertRaisesRegex(RuntimeError,'Unit tests failed'):
                    verify_windows.main()
            result=json.loads((root/'outputs/windows_native_verification/status.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['native_kernel_verified'])

    def test_copy_rejects_workspace_inside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                verify_windows.copy_project(temporary,Path(temporary)/'nested')

    def test_non_windows_cannot_report_pass(self):
        with patch.object(sys,'argv',['verify_windows']), patch.object(verify_windows.platform,'system',return_value='Linux'):
            with self.assertRaises(SystemExit) as caught:
                verify_windows.main()
            self.assertIn('requires actual Windows',str(caught.exception))


if __name__=='__main__':
    unittest.main()
