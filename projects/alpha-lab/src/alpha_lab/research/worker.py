"""Trusted CPU-only entry point; no arbitrary plugin, provider or dotenv setup."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.market_neutral import MarketNeutralBacktest
from .bundles import load_bundle
from .contracts import ResearchError, ResearchSpec, parse, read_json
from .runtime import environment_identity, implementation_identity
from .workflow import compatible


def evaluate(bundle_path, spec_path, output):
    bundle, spec = load_bundle(bundle_path), parse(ResearchSpec, read_json(spec_path))
    compatible(spec, bundle)
    if (os.environ.get('OMP_NUM_THREADS') != str(spec.threads)
            or os.environ.get('OPENBLAS_NUM_THREADS') != str(spec.threads)
            or os.environ.get('PYTHONHASHSEED') != str(spec.seed)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ResearchError('Worker numerical environment is inconsistent')
    provider_keys = any('KEY' in name.upper() or 'TOKEN' in name.upper() or 'SECRET' in name.upper()
                        for name in os.environ)
    proxies = any('PROXY' in name.upper() for name in os.environ)
    if provider_keys or proxies:
        raise ResearchError('Compute worker received credentials or proxy settings')
    np.random.seed(spec.seed)
    engine = MarketNeutralBacktest(Path(bundle_path))
    # All underlying files have already been materialized through <=2022.
    result, daily = engine.evaluate(spec.strategy.model_dump(), include_holdout=False,
                                   cost_bps=spec.cost_bps, borrow_bps=spec.borrow_bps_annual,
                                   terminal_stress=spec.terminal_stress)
    output = Path(output)
    if any(output.iterdir()):
        raise ResearchError('Worker output must be empty')
    atomic_json(output/'metrics.json', {'kind': 'provisional_legacy_development_result',
                                      'spec_id': spec.semantic_id(), 'bundle_id': bundle.bundle_id,
                                      'legacy_result': result})
    daily.to_csv(output/'daily.csv', index=False)
    atomic_json(output/'runtime.json', {
        'numerical_policy': spec.numerical_policy, 'threads': spec.threads, 'seed': spec.seed,
        'provider_keys_present': provider_keys, 'proxy_present': proxies,
        'environment': environment_identity(), 'implementation': implementation_identity(),
        'isolation': 'credential_filtered_process_not_filesystem_or_network_sandbox',
        'resource_guards': {'timeout_seconds': spec.timeout_seconds, 'max_output_bytes': spec.max_output_bytes,
                            'sampled_rss_limit_mb': spec.memory_limit_mb, 'hard_os_memory_limit': False},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'spec', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args.bundle, args.spec, args.output)
    except ResearchError as error:
        print('Local research worker failed: '+str(error), flush=True)
        return 1
    except Exception as error:
        # Exceptions may contain data/path values; preserve only a safe outcome.
        print('Local research worker failed: '+type(error).__name__+'; no complete result was published', flush=True)
        return 1
    print('Local development evaluation finished', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
