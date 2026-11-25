from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / 'runs' / 'pilot-20261118-184624-535'

def run(folder, script, *args):
    print('Running', script, *args, flush=True)
    subprocess.run([sys.executable, str(folder / script), *args], cwd=folder, check=True)

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    PILOT.mkdir(parents=True, exist_ok=True)
    for name in ['download_data.py','run_experiment.py','evaluate.py','verify_results.py','config.json']:
        shutil.copy2(ROOT / name, PILOT / name)
    shutil.copytree(ROOT / 'data' / 'raw', PILOT / 'data' / 'raw', dirs_exist_ok=True)
    (ROOT / 'report').mkdir(exist_ok=True)
    run(PILOT, 'download_data.py')
    run(PILOT, 'run_experiment.py', '--stage', 'check')
    for horizon in ['5','22']:
        for stage in ['validate','test']:
            run(PILOT, 'run_experiment.py', '--stage', stage, '--horizon', horizon)
    run(PILOT,'evaluate.py')
    run(PILOT,'verify_results.py')
    for stage in ['validate','lock','explore','confirm']:
        run(ROOT,'main_experiment.py',stage)
    for script in ['verify_main.py','write_main_report.py','plot_main_intervals.py']:
        run(ROOT,script)
