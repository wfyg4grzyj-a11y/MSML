"""Standalone experiment entry point, usable by any configured Python environment."""

import runpy
import sys
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
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        raise
    except BaseException:
        code = 1
        raise
    finally:
        Path('local_exit_code.txt').write_text(str(code), encoding='utf-8')
