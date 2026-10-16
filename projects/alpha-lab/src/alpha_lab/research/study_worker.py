"""Fixed raw-accounting worker; no generated evaluator, provider or secret setup."""
import argparse
import os
from pathlib import Path

from alpha_lab.market_data.sharadar_store import atomic_json
from .contracts import parse, read_json
from .study_contracts import StudySpec
from .study_engine import evaluate_study
from .study_inputs import load_study
from .study_workflow import study_environment, validate_study_spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','spec','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        bundle,packet=load_study(args.bundle)
        spec=parse(StudySpec,read_json(args.spec))
        validate_study_spec(spec,bundle)
        ledger,portfolio=evaluate_study(packet,args.bundle)
        atomic_json(args.output/'ledger.json',ledger)
        atomic_json(args.output/'portfolio.json',portfolio)
        atomic_json(args.output/'runtime.json',{'environment':study_environment(),'implementation':spec.implementation,
                    'threads':spec.threads,'seed':spec.seed,'provider_keys_present':any(k.upper().endswith(('API_KEY','ACCESS_TOKEN','SECRET_ACCESS_KEY')) for k in os.environ),
                    'proxy_present':any(k.upper().endswith('_PROXY') for k in os.environ),
                    'isolation':'credential_filtered_process_not_filesystem_or_network_sandbox'})
        return 0
    except Exception:
        print('Study worker failed; require preserved run evidence, no automatic retry',flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
