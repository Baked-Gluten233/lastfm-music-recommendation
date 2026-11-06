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

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true',help='List platform and dependency availability without running acceptance checks')
    args=parser.parse_args()
    result={'platform':platform.platform(),'python':sys.version,'executable':sys.executable,
            'native_windows':platform.system()=='Windows','pointer_bits':64 if sys.maxsize>2**32 else 32,
            'started_at_utc':datetime.now(timezone.utc).isoformat(),'native_kernel_verified':False,
            'final_test_rescored':False,'status':'not_run'}
    packages={}
    from importlib.metadata import version,PackageNotFoundError
    for name in ['numpy','scipy','pandas','matplotlib','torch','nbformat','nbclient','ipykernel']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    result['packages']=packages
    if args.preflight:
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if not result['native_windows']:
        raise SystemExit('This acceptance runner requires actual Windows; no Windows pass has been recorded.')
    if result['pointer_bits']!=64 or sys.version_info[:2]!=(3,12) or any(v is None for v in packages.values()):
        raise SystemExit('Use a 64-bit Python 3.12 virtual environment with base and phase2 requirements installed.')
    import nbformat
    from nbclient import NotebookClient
    output=ROOT/'outputs/windows_native_verification';output.mkdir(parents=True,exist_ok=True)
    status_path=output/'status.json'
    def save():status_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    try:
        result['status']='running';save()
        process=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
        (output/'unit_tests.log').write_text(process.stdout+process.stderr,encoding='utf-8',newline='\n')
        if process.returncode:raise RuntimeError('Unit tests failed; inspect unit_tests.log')
        result['unit_tests']='passed';result['notebooks']=[];save()
        env=os.environ.copy()
        env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
        env['JUPYTER_PATH']=str(Path(sys.prefix)/'share/jupyter')+os.pathsep+env.get('JUPYTER_PATH','')
        env['IPYTHONDIR']=str(output/'ipython')
        env['JUPYTER_RUNTIME_DIR']=str(output/'runtime')
        for filename in ['01_data_and_baselines.ipynb','02_model_iterations.ipynb','03_frozen_final_test.ipynb']:
            notebook=nbformat.read(ROOT/'notebooks'/filename,as_version=4)
            # Copy only: do not replace the shipped evidence notebooks.
            for cell in notebook.cells:
                if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
            check=nbformat.v4.new_code_cell(f"import sys\nfrom pathlib import Path\nassert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve(), 'Wrong Python kernel'")
            notebook.cells.insert(0,check)
            NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute(env=env)
            nbformat.validate(notebook)
            nbformat.write(notebook,output/filename)
            result['notebooks'].append({'name':filename,'native_kernel':'passed'});save()
        result.update(status='passed',native_kernel_verified=True,completed_at_utc=datetime.now(timezone.utc).isoformat());save()
        print('Native Windows tests and three Jupyter notebook executions passed. See',status_path)
    except Exception as error:
        result.update(status='failed',error=str(error));save();raise

if __name__=='__main__':main()
