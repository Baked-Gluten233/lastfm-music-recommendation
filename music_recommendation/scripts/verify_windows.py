"""Native Windows acceptance checks; never marks Linux as Windows tested.

Runs tests and all notebooks in actual Jupyter kernels. Final notebook only
verifies saved final artifacts, so this does not repeat held-out test scoring.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import platform
import subprocess
import sys
import shutil
import tempfile
import hashlib
import re
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


NOTEBOOKS = {
    '01_data_and_baselines.ipynb': 'create_notebook.py',
    '02_model_iterations.ipynb': 'create_phase2_notebook.py',
    '03_frozen_final_test.ipynb': 'create_final_notebook.py',
}
EXCLUDED = {'.venv', 'venv', '__pycache__', '.git', '.cache', '.mplconfig',
            '.ipynb_checkpoints', 'deliverables', 'windows_native_verification',
            'content_feature_audit'}
FINAL_TEST_GUARD = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
import musicrec.final_evaluation as _acceptance_final
def _reject_final_test_scoring(*args, **kwargs):
    raise RuntimeError('Held-out test scoring is disabled during acceptance')
_acceptance_final.run_final_once = _reject_final_test_scoring
_acceptance_final.evaluate = _reject_final_test_scoring
print('Acceptance guard installed for final test scoring')
"""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2),
                          encoding='utf-8', newline='\n')


def validate_temp_directory():
    """Reject a configured temporary path that would silently fall back."""
    requested = next((os.environ[name] for name in ('TMPDIR', 'TEMP', 'TMP')
                      if os.environ.get(name)), None)
    if requested is not None:
        configured = Path(requested).resolve()
        try:
            with tempfile.TemporaryFile(dir=configured):
                pass
        except OSError as error:
            raise RuntimeError('Configured temporary directory is unavailable: ' + str(configured)) from error
        actual = Path(tempfile.gettempdir()).resolve()
        if actual != configured:
            raise RuntimeError('Temporary directory fallback detected: ' + str(actual))
        return str(actual)
    return tempfile.gettempdir()


def file_hashes(root, directories):
    """Protect file names and bytes, excluding caches and independent new runs."""
    root = Path(root)
    result = {}
    for name in directories:
        for path in sorted((root / name).rglob('*')):
            relative = path.relative_to(root)
            if any(part in EXCLUDED for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError('Acceptance inputs must not be symlinks: ' + str(relative))
            if path.is_file():
                result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def manifest_digest(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def source_manifest(root):
    manifest = file_hashes(root, ('src', 'scripts', 'tests'))
    for name in ('requirements.txt', 'requirements-phase2.txt'):
        manifest[name] = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
    return manifest


def integrity_result(before, after):
    return {
        'passed': before == after,
        'files_before': len(before), 'files_after': len(after),
        'before_sha256': manifest_digest(before), 'after_sha256': manifest_digest(after),
        'added': sorted(after.keys() - before.keys()),
        'removed': sorted(before.keys() - after.keys()),
        'changed': sorted(name for name in before.keys() & after.keys()
                          if before[name] != after[name]),
    }


def run_command(command, work, env, log_path):
    started = time.perf_counter()
    process = subprocess.run(command, cwd=work, env=env, capture_output=True,
                             text=True, encoding='utf-8')
    log = process.stdout + process.stderr
    Path(log_path).write_text(log, encoding='utf-8', newline='\n')
    return process, {'command': command, 'exit_code': process.returncode,
                     'duration_seconds': time.perf_counter() - started,
                     'log': Path(log_path).name}, log


def run_tests(source, output, env, optimized):
    name = 'optimized' if optimized else 'normal'
    command = [sys.executable] + (['-O'] if optimized else [])
    command += ['-m', 'unittest', 'discover', '-s', 'tests', '-v']
    log_path = Path(output) / ('unit_tests_' + name + '.log')
    with tempfile.TemporaryDirectory(prefix='musicrec_tests_' + name + '_') as temporary:
        work = copy_project(source, Path(temporary) / 'project')
        process, summary, log = run_command(command, work, env, log_path)
    count = re.search(r'^Ran (\d+) tests? in ', log, re.MULTILINE)
    skipped = re.search(r'\bskipped=(\d+)', log)
    summary.update(tests=int(count.group(1)) if count else 0,
                   skipped=int(skipped.group(1)) if skipped else 0)
    summary['passed'] = process.returncode == 0 and summary['tests'] > 0 and summary['skipped'] == 0
    return summary


def verify_dependency_pins(root, packages):
    for filename in ('requirements.txt', 'requirements-phase2.txt'):
        for line in (Path(root) / filename).read_text(encoding='utf-8').splitlines():
            if '==' in line and not line.lstrip().startswith('#'):
                name, expected = line.strip().split('==', 1)
                installed = packages.get(name)
                if installed is None or installed.split('+', 1)[0] != expected:
                    raise RuntimeError('Pinned dependency mismatch: ' + name)


def copy_project(source, target):
    """Copy only runtime inputs, never links or prior acceptance workspaces.

    Each notebook gets its OWN copy of the frozen snapshot: phase 1 may rewrite
    validation evidence and must not contaminate the final notebook's checks.
    This isolates cooperative project code, not arbitrary untrusted code.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError('Acceptance workspace must be outside the source project')
    target.mkdir(parents=True, exist_ok=True)
    def ignore(directory, names):
        return [name for name in names if name in EXCLUDED or
                name.endswith(('.pyc', '.pyo')) or
                (Path(directory) / name).is_symlink()]
    for name in ('src', 'scripts', 'tests', 'data', 'outputs'):
        path = source / name
        if path.is_symlink():
            raise ValueError('Runtime input must not be a symlink: ' + name)
        if path.is_dir():
            shutil.copytree(path, target / name, ignore=ignore)
    for name in ('requirements.txt', 'requirements-phase2.txt'):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    (target / 'notebooks').mkdir(exist_ok=True)
    return target


def execute_notebook(source, output, filename, env, client_class=None):
    """Regenerate a fresh source notebook and execute only in a disposable copy."""
    import nbformat
    from nbclient import NotebookClient
    if filename not in NOTEBOOKS:
        raise ValueError('Unknown acceptance notebook')
    client_class = client_class or NotebookClient
    started = time.perf_counter()
    notebook = None
    log_path = Path(output) / (filename + '.log')
    try:
        with tempfile.TemporaryDirectory(prefix='musicrec_acceptance_') as temporary:
            work = copy_project(source, Path(temporary) / 'project')
            subprocess.run([sys.executable, str(work / 'scripts' / NOTEBOOKS[filename])],
                           cwd=work, env=env, check=True, capture_output=True,
                           text=True, encoding='utf-8')
            notebook = nbformat.read(work / 'notebooks' / filename, as_version=4)
            check = nbformat.v4.new_code_cell(
                "import sys\nfrom pathlib import Path\n"
                f"if Path(sys.executable).resolve() != Path({sys.executable!r}).resolve():\n"
                "    raise RuntimeError('Wrong Python kernel')")
            notebook.cells[:0] = [check, nbformat.v4.new_code_cell(FINAL_TEST_GUARD)]
            client_class(notebook, timeout=600, kernel_name='python3',
                         resources={'metadata': {'path': str(work)}}).execute(env=env)
            nbformat.validate(notebook)
        summary = {'name': filename, 'native_kernel': 'passed', 'final_test_guarded': True,
                   'duration_seconds': time.perf_counter() - started,
                   'code_cells': sum(cell.cell_type == 'code' for cell in notebook.cells),
                   'error_outputs': sum(item.output_type == 'error' for cell in notebook.cells
                                        for item in cell.get('outputs', [])),
                   'log': log_path.name}
        if summary['error_outputs']:
            raise RuntimeError('Notebook contains error outputs')
        log_path.write_text(json.dumps(summary, indent=2), encoding='utf-8', newline='\n')
        return summary
    except Exception as error:
        details=traceback.format_exc()
        if isinstance(error,subprocess.CalledProcessError):
            details+='\n'+(error.stdout or '')+(error.stderr or '')
        log_path.write_text(details, encoding='utf-8', newline='\n')
        raise
    finally:
        if notebook is not None:
            nbformat.write(notebook, Path(output) / filename)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true',help='List platform and dependency availability without running acceptance checks')
    args=parser.parse_args()
    started = time.perf_counter()
    result={'platform':platform.platform(),'python':sys.version,'executable':sys.executable,
            'native_windows':platform.system()=='Windows','pointer_bits':64 if sys.maxsize>2**32 else 32,
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'native_kernel_verified':False,
            'final_test_rescored':False,'isolated_per_notebook':True,'status':'not_run',
            'test_suites':{},'notebooks':[],'temporary_directory':None}
    packages={}
    from importlib.metadata import version,PackageNotFoundError
    for name in ['numpy','scipy','pandas','matplotlib','torch','nbformat','nbclient','ipykernel','jupyterlab']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    result['packages']=packages
    if args.preflight:
        result['temporary_directory']=validate_temp_directory()
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    output=ROOT/'outputs/windows_native_verification';output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    before = None
    failure = None
    try:
        result['status']='running';write_json(status_path,result)
        before = file_hashes(ROOT, ('data','notebooks','outputs'))
        write_json(output/'protected_files_before.json', before)
        source = source_manifest(ROOT)
        result['source'] = {'sha256':manifest_digest(source),'files':source}
        result['temporary_directory']=validate_temp_directory()
        if not result['native_windows']:
            raise SystemExit('This acceptance runner requires actual Windows; no Windows pass has been recorded.')
        if result['pointer_bits']!=64 or sys.version_info[:2]!=(3,12) or any(v is None for v in packages.values()):
            raise RuntimeError('Use a 64-bit Python 3.12 virtual environment with base and phase2 requirements installed.')
        verify_dependency_pins(ROOT, packages)
        env=os.environ.copy()
        env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
        env['JUPYTER_PATH']=str(Path(sys.prefix)/'share/jupyter')+os.pathsep+env.get('JUPYTER_PATH','')
        env['IPYTHONDIR']=str(output/'ipython')
        env['JUPYTER_RUNTIME_DIR']=str(output/'runtime')
        process,result['pip_check'],_=run_command(
            [sys.executable,'-m','pip','check'],ROOT,env,output/'pip_check.log')
        if process.returncode:
            raise RuntimeError('Dependency check failed; inspect pip_check.log')
        for optimized in (False,True):
            name='optimized' if optimized else 'normal'
            summary=run_tests(ROOT,output,env,optimized)
            result['test_suites'][name]=summary
            write_json(status_path,result)
            if not summary['passed']:
                raise RuntimeError('Unit tests failed or skipped: '+name+'; inspect '+summary['log'])
        result['unit_tests']='passed'
        result['optimized_unit_tests']='passed'
        for filename in NOTEBOOKS:
            notebook_started=time.perf_counter()
            try:
                result['notebooks'].append(execute_notebook(ROOT, output, filename, env))
            except Exception as error:
                result['notebooks'].append({'name':filename,'native_kernel':'failed',
                    'duration_seconds':time.perf_counter()-notebook_started,
                    'log':filename+'.log','error':str(error)})
                raise
            write_json(status_path,result)
        result['native_kernel_verified']=True
    except BaseException as error:
        failure=error
        result.update(status='failed',error=str(error))
        (output/'failure.log').write_text(traceback.format_exc(),encoding='utf-8',newline='\n')
    finally:
        try:
            if before is not None:
                after=file_hashes(ROOT,('data','notebooks','outputs'))
                write_json(output/'protected_files_after.json',after)
                result['integrity']=integrity_result(before,after)
                if not result['integrity']['passed']:
                    raise RuntimeError('Frozen input or output files changed during acceptance')
            if 'source' in result:
                result['source_unchanged']=source_manifest(ROOT)==result['source']['files']
                if not result['source_unchanged']:
                    raise RuntimeError('Runtime source changed during acceptance')
        except Exception as error:
            result['integrity_error']=str(error)
            if failure is None:failure=error
        result.update(status='failed' if failure else 'passed',
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      duration_seconds=time.perf_counter()-started)
        write_json(status_path,result)
    if failure is not None:
        raise failure
    print('Native Windows tests and three Jupyter notebook executions passed. See',status_path)

if __name__=='__main__':main()
