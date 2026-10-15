"""Pin -> resolve spec -> run -> inspect; entirely local and development-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

import duckdb

from .bundles import load_bundle, pin_panel
from .contracts import ResearchError, ResearchSpec, parse, read_json
from .workflow import compatible, inspect_run, make_spec, run_research, validate_dependencies


def signal_commands(commands):
    init = commands.add_parser('registry-init', help='Create a new immutable-schema local trial ledger')
    init.add_argument('path', type=Path)
    show = commands.add_parser('registry-show', help='Read-only project-wide trials, attempts and exposure history')
    show.add_argument('path', type=Path)
    exposure = commands.add_parser('registry-exposure', help='Record previous knowledge using a hash-verified evidence file')
    exposure.add_argument('--registry', type=Path, required=True)
    exposure.add_argument('--record', type=Path, required=True)
    exposure.add_argument('--evidence', type=Path, required=True)
    commands.add_parser('signal-catalog', help='Show independent feature definitions and fixed recipes')
    spec = commands.add_parser('signal-spec', help='Resolve an explicit development signal request')
    spec.add_argument('--request', type=Path, required=True)
    spec.add_argument('--bundle', type=Path, required=True)
    spec.add_argument('--output', type=Path, required=True)
    for name in ('signal-register', 'signal-check', 'signal-run'):
        child = commands.add_parser(name)
        child.add_argument('--bundle', type=Path, required=True)
        child.add_argument('--spec', type=Path, required=True)
        if name != 'signal-check':
            child.add_argument('--registry', type=Path, required=True)
        if name == 'signal-register':
            child.add_argument('--family', type=Path, required=True)
            child.add_argument('--hypothesis', type=Path, required=True)
            child.add_argument('--parent')
        if name == 'signal-run':
            child.add_argument('--output', type=Path, required=True)
            child.add_argument('--cache', type=Path, required=True)
            child.add_argument('--retry-of')
    inspect = commands.add_parser('signal-inspect', help='Read-only archived artifact verification')
    inspect.add_argument('directory', type=Path)
    reconcile = commands.add_parser('signal-reconcile', help='Explicitly reconcile validated files with registered attempt')
    reconcile.add_argument('directory', type=Path)
    reconcile.add_argument('--registry', type=Path, required=True)
    compare = commands.add_parser('signal-compare', help='Read-only compatible development diagnostics; no selection')
    compare.add_argument('directories', nargs='+', type=Path)


def signal_action(args):
    from .features import feature_definition
    from .registry import Registry, initialize_registry
    from .signal_contracts import FEATURES, Family, Hypothesis, PriorExposure, SignalSpec, baseline_recipes
    from .signal_workflow import (compare_signals, inspect_signal, reconcile_signal, resolve_signal,
                                  run_signal, validate_signal)
    if args.command == 'registry-init':
        initialize_registry(args.path)
        return {'registry': str(args.path), 'schema_version': 2}
    if args.command == 'registry-show':
        with Registry(args.path, readonly=True) as registry:
            return registry.snapshot()
    if args.command == 'registry-exposure':
        exposure = parse(PriorExposure, read_json(args.record))
        with Registry(args.registry) as registry:
            registry.note_prior_exposure(exposure, args.evidence)
        return {'exposure_id': exposure.exposure_id, 'recorded': True}
    if args.command == 'signal-catalog':
        return {'features': [feature_definition(n) for n in FEATURES], 'recipes': baseline_recipes()}
    if args.command == 'signal-spec':
        spec = resolve_signal(args.bundle, read_json(args.request))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as target:
            json.dump(spec.model_dump(), target, indent=2, allow_nan=False)
        return {'trial_id': spec.semantic_id(), 'comparison_id': spec.comparison_id(), 'output': str(args.output)}
    if args.command == 'signal-inspect':
        return inspect_signal(args.directory)
    if args.command == 'signal-reconcile':
        return reconcile_signal(args.registry, args.directory)
    if args.command == 'signal-compare':
        return compare_signals(args.directories)
    if args.command == 'signal-run':
        return run_signal(args.bundle, args.spec, args.output, args.cache, args.registry, retry_of=args.retry_of)
    spec = parse(SignalSpec, read_json(args.spec))
    validate_signal(spec, load_bundle(args.bundle), args.bundle)
    if args.command == 'signal-check':
        return {'development_signal_ready': True, 'trial_id': spec.semantic_id(), 'holdout_access': False, 'llm_calls': 0}
    with Registry(args.registry) as registry:
        trial = registry.register(parse(Family, read_json(args.family)), parse(Hypothesis, read_json(args.hypothesis)), spec, args.parent)
    return {'trial_id': trial, 'registered': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pin = commands.add_parser('pin', help='Materialize schema-4 panel through 2022 at the latest')
    pin.add_argument('--panel', type=Path, required=True)
    pin.add_argument('--output', type=Path, required=True)
    pin.add_argument('--end-date', required=True)
    spec = commands.add_parser('spec', help='Resolve explicit legacy research/numerical/resource policies')
    spec.add_argument('--bundle', type=Path, required=True)
    spec.add_argument('--strategy', type=Path, required=True)
    spec.add_argument('--output', type=Path, required=True)
    spec.add_argument('--hypothesis-id', required=True)
    spec.add_argument('--family-id', required=True)
    spec.add_argument('--title', required=True)
    for name in ('cost-bps', 'borrow-bps-annual', 'terminal-stress'):
        spec.add_argument('--'+name, type=float, required=True)
    for name in ('seed', 'threads', 'timeout-seconds', 'max-output-bytes', 'memory-limit-mb'):
        spec.add_argument('--'+name, type=int, required=True)
    for name in ('run', 'check'):
        child = commands.add_parser(name, help='Execute locally' if name == 'run' else 'Read-only validation; no execution')
        child.add_argument('--bundle', type=Path, required=True)
        child.add_argument('--spec', type=Path, required=True)
        if name == 'run':
            child.add_argument('--output', type=Path, required=True)
    inspect = commands.add_parser('inspect', help='Read-only artifact verification; never retries or starts APIs')
    inspect.add_argument('directory', type=Path)
    signal_commands(commands)
    from .product_cli import product_commands
    product_commands(commands)
    args = parser.parse_args(argv)
    try:
        if args.command.startswith(('study-', 'risk-', 'research-', 'evidence-')) or args.command=='public-design':
            from .product_cli import product_action
            result=product_action(args)
        elif args.command.startswith(('signal-', 'registry-')):
            result = signal_action(args)
        elif args.command == 'pin':
            result = pin_panel(args.panel, args.output, end=args.end_date).model_dump()
        elif args.command == 'spec':
            values = vars(args).copy()
            for name in ('command', 'output', 'strategy'):
                values.pop(name)
            resolved = make_spec(**values, strategy=read_json(args.strategy))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as target:
                json.dump(resolved.model_dump(), target, indent=2, allow_nan=False)
            result = {'spec_id': resolved.semantic_id(), 'output': str(args.output)}
        elif args.command == 'run':
            result = run_research(args.bundle, args.spec, args.output).model_dump()
        elif args.command == 'check':
            bundle = load_bundle(args.bundle)
            resolved = parse(ResearchSpec, read_json(args.spec))
            compatible(resolved, bundle)
            validate_dependencies(args.bundle, resolved)
            result = {'ready_for_local_legacy_execution': True, 'spec_id': resolved.semantic_id(),
                      'bundle_id': bundle.bundle_id, 'llm_calls': 0, 'holdout_access': False,
                      'scientific_disposition': 'provisional_legacy_research_not_accepted'}
        else:
            result = inspect_run(args.directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except ResearchError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error, sqlite3.Error):
        print('Cannot complete local research operation; check settings, paths and saved evidence', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
