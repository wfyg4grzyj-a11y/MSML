"""Trusted fixed signal computation; no user code, providers or registry mutations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import load_bundle
from .contracts import ResearchError, parse, read_json
from .features import KEYS, cached_feature, write_parquet
from .runtime import environment_identity, implementation_identity
from .signal_contracts import SignalSpec
from .signals import evaluate_signal, make_labels
from .workflow import compatible


def compute(bundle_path, spec, output, cache):
    bundle = load_bundle(bundle_path)
    compatible(spec, bundle)
    frame, feature_ids = None, {}
    for name in spec.recipe.computation()['factors']:
        raw, key = cached_feature(bundle_path, bundle, name, cache, spec.max_output_bytes)
        feature_ids[name] = key
        if frame is None:
            frame = raw[KEYS + ['source_member']].copy()
        elif not frame[KEYS].equals(raw[KEYS]):
            raise ResearchError('Feature artifacts disagree on source membership')
        for source, target in [('value', 'raw_'), ('feature_available', 'available_'),
                               ('availability_reason', 'reason_'), ('information_date', 'information_')]:
            frame[target+name] = raw[source]
    with np.load(Path(bundle_path)/'returns.npz', allow_pickle=False) as data:
        arrays = {n: data[n] for n in data.files}
    # Cache population is reusable across the bounded bundle, but calculate
    # outcomes only for decisions that belong to this preregistered study.
    selected = frame.date.eq(None)
    for fold in spec.folds:
        selected |= frame.date.between(fold.train_start, fold.train_end) | frame.date.between(fold.validation_start, fold.validation_end)
    frame = frame.loc[selected].reset_index(drop=True)
    labels = make_labels(frame, arrays, spec.label)
    frame = frame.merge(labels, on=KEYS, validate='one_to_one')
    predictions, daily, quantiles, fits, summary = evaluate_signal(frame, spec, arrays['dates'])
    for name, table in [('predictions', predictions), ('daily', daily), ('quantiles', quantiles)]:
        write_parquet(table, output/(name+'.parquet'))
    atomic_json(output/'fits.json', fits)
    atomic_json(output/'metrics.json', {'kind': 'development_signal_diagnostics_v1',
                'trial_id': spec.semantic_id(), 'comparison_id': spec.comparison_id(), 'bundle_id': bundle.bundle_id,
                'feature_ids': feature_ids, 'folds': summary, 'investment_ready': False,
                'point_in_time_attested': False, 'rows': len(predictions), 'llm_calls': 0})
    env = {k.upper() for k in os.environ}
    atomic_json(output/'runtime.json', {'environment': environment_identity(), 'implementation': implementation_identity(),
                'threads': spec.threads, 'seed': spec.seed, 'numerical_policy': spec.numerical_policy,
                'provider_keys_present': any(k.endswith(('API_KEY', 'ACCESS_TOKEN', 'SECRET_ACCESS_KEY')) for k in env),
                'proxy_present': any(k.endswith('_PROXY') for k in env),
                'isolation': 'credential_filtered_process_not_filesystem_or_network_sandbox'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'spec', 'output', 'cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        compute(args.bundle, parse(SignalSpec, read_json(args.spec)), args.output, args.cache)
        return 0
    except Exception:
        # Third-party error strings may contain local data; retain only a fixed code.
        print('Signal worker failed; no completion published', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
