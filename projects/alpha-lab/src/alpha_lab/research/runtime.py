"""Explicit CPU worker environment and implementation/environment identities."""
from __future__ import annotations

from importlib.metadata import version
import os
from pathlib import Path
import platform

from .contracts import file_hash

SYSTEM_ENV = {'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'PATH',
              'PROCESSOR_ARCHITECTURE', 'PROCESSOR_ARCHITEW6432'}


def implementation_identity():
    package = Path(__file__).resolve().parents[1]
    names = [str(p.relative_to(package)).replace('\\', '/') for p in sorted((package/'research').glob('*.py'))]
    names += ['market_neutral.py', 'equity_calendar.py', 'local_process.py',
              'market_data/sharadar.py', 'market_data/sharadar_store.py', 'market_data/sharadar_cli.py']
    return {name: file_hash(package/name) for name in names}


def environment_identity():
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'os': platform.system(), 'machine': platform.machine(),
            **{name: version(name) for name in ('numpy', 'pandas', 'duckdb', 'exchange-calendars', 'pydantic')}}


def child_environment(spec):
    env = {key: value for key, value in os.environ.items() if key.upper() in SYSTEM_ENV}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
               PYTHONHASHSEED=str(spec.seed), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS=str(spec.threads), OPENBLAS_NUM_THREADS=str(spec.threads),
               MKL_NUM_THREADS=str(spec.threads), NUMEXPR_NUM_THREADS=str(spec.threads),
               VECLIB_MAXIMUM_THREADS=str(spec.threads), RAYON_NUM_THREADS=str(spec.threads))
    return env
