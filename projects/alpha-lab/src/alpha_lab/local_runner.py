"""Standalone experiment entry point, usable by any configured Python environment."""

import runpy
import sys
import json
import os
import re
import tempfile
import time
from pathlib import Path


def run(gpu: bool) -> None:
    if gpu:
        import torch
        original = torch.use_deterministic_algorithms

        def patched(mode=True, **kwargs):
            if mode:
                print('[Local GPU wrapper] Intercepted torch.use_deterministic_algorithms(True) -> skipped')
                return
            original(mode, **kwargs)

        torch.use_deterministic_algorithms = patched
        try:
            import lightning.pytorch as pl
        except Exception:
            pass
        else:
            original_init = pl.Trainer.__init__

            def trainer_init(self, *args, deterministic=None, **kwargs):
                if deterministic:
                    deterministic = False
                original_init(self, *args, deterministic=deterministic, **kwargs)

            pl.Trainer.__init__ = trainer_init
    sys.argv = ['run_experiment.py']
    sys.path.insert(0, str(Path.cwd()))
    runpy.run_path('run_experiment.py', run_name='__main__')


if __name__ == '__main__':
    gpu = sys.argv[1] == 'gpu'
    code = 0
    try:
        run(gpu)
    except SystemExit as error:
        code = int(error.code) if isinstance(error.code, int) else (0 if error.code is None else 1)
        raise
    except BaseException:
        code = 1
        raise
    finally:
        Path('local_exit_code.txt').write_text(str(code), encoding='utf-8')
        job_id = os.environ.get('ALPHALAB_JOB_ID', '')
        if re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job_id):
            receipt = {'schema_version': 1, 'job_id': job_id,
                       'resource': 'gpu' if gpu else 'cpu', 'exit_code': code,
                       'finished_at': time.time()}
            absolute = os.path.abspath(f'local_job_result-{job_id}.json')
            if os.name == 'nt' and not absolute.startswith('\\\\?\\'):
                absolute = ('\\\\?\\UNC\\'+absolute[2:]) if absolute.startswith('\\\\') else '\\\\?\\'+absolute
            path = Path(absolute)
            descriptor, temporary = tempfile.mkstemp(prefix='job.', suffix='.tmp', dir=path.parent)
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                    json.dump(receipt, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
