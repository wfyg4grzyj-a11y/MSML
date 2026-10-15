"""Pin -> resolve spec -> run -> inspect; entirely local and development-only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import duckdb

from .bundles import load_bundle, pin_panel
from .contracts import ResearchError, ResearchSpec, parse, read_json
from .workflow import compatible, inspect_run, make_spec, run_research, validate_dependencies


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
    args = parser.parse_args(argv)
    try:
        if args.command == 'pin':
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
    except (OSError, ValueError, KeyError, TypeError, duckdb.Error):
        print('Cannot complete local research operation; check settings, paths and saved evidence', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
