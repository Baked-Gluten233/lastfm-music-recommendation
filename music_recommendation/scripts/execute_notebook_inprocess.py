"""Execute real notebook cells in order without a socket-based Jupyter kernel.

Used only where local kernel sockets are unavailable. This is NOT a substitute
for a native Jupyter Restart-and-Run-All acceptance test on the target platform.
No outputs are invented: stdout, stderr and rich displays come from execution.
"""
from pathlib import Path
import os
import argparse
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--notebook',default='01_data_and_baselines.ipynb')
    args=parser.parse_args()
    os.chdir(ROOT)
    path = ROOT / 'notebooks' / args.notebook
    if not path.resolve().is_relative_to((ROOT/'notebooks').resolve()):
        raise ValueError('Notebook must be inside project notebooks directory')
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Cell {count} failed: {result.error_before_exec or result.error_in_exec}')
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        cell.outputs = outputs
        cell.execution_count = count
    notebook.metadata['verification'] = {
        'execution_method': 'IPython in-process sequential cells, real captured outputs',
        'native_jupyter_kernel_verified': False,
        'windows_verified': False,
        'reason': 'Native Jupyter kernel could not start because local socket/network-interface operations are unavailable in the execution environment',
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    images = sum('image/png' in output.get('data', {}) for cell in notebook.cells
                 if cell.cell_type == 'code' for output in cell.outputs)
    print(f'Executed {count} real code cells in sequence; captured {images} PNG displays; schema valid.')
    print('Native Jupyter kernel / Windows execution: NOT VERIFIED.')

if __name__ == '__main__':
    main()
