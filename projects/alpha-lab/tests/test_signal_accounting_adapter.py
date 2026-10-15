"""P2 outcomes are not a filter when compiling provisional accounting inputs."""
import pytest
pytest.importorskip('numpy')
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.contracts import ResearchError,read_json
from alpha_lab.research.registry import Registry,initialize_registry
from alpha_lab.research.signal_adapter import packet_from_signal
from alpha_lab.research.signal_workflow import resolve_signal,run_signal
from tests.test_signal_lab import signal_bundle,request,registration
from tests.test_integrated_study import packet
from tests.test_accounting_ledger import scenario


def test_bridge_keeps_censored_scores_and_requires_explicit_issuer_policy(signal_bundle,tmp_path):
    spec=resolve_signal(signal_bundle,request());source=tmp_path/'signal-spec.json';atomic_json(source,spec.model_dump())
    ledger=tmp_path/'registry.sqlite';initialize_registry(ledger)
    with Registry(ledger) as db:registration(db,spec)
    run_signal(signal_bundle,source,tmp_path/'signal-run',tmp_path/'cache',ledger)
    value=packet();value['contexts']=[]
    raw=scenario((100.,100.,100.),start='2014-05-29',end='2014-06-02')
    raw.update(history_mode='public_information_reconstruction',point_in_time_attested=False,
               securities={str(i+1):f'S{i:02d}' for i in range(4)})
    for session in raw['sessions']:session['quotes']={sid:dict(session['quotes']['A']) for sid in raw['securities']}
    value.update(scenario=raw,identity_namespace='sharadar_permaticker_share_class')
    value['portfolio_policy'].update(max_issuer_gross=None,max_sector_net=None)
    template=tmp_path/'template.json';atomic_json(template,value)
    result=packet_from_signal(tmp_path/'signal-run',signal_bundle,template,tmp_path/'compiled.json')
    assert result['future_outcome_filter'] is False and result['point_in_time_attested'] is False
    compiled=read_json(tmp_path/'compiled.json')
    assert compiled['contexts'][0]['date']=='2014-05-30'
    assert all(p['score'] is not None for p in compiled['contexts'][0]['points'])
    assert all(p['issuer_id'] is None for p in compiled['contexts'][0]['points'])
    value['portfolio_policy']['max_issuer_gross']=.2;atomic_json(template,value)
    with pytest.raises(ResearchError,match='issuer'):packet_from_signal(tmp_path/'signal-run',signal_bundle,template,tmp_path/'blocked.json')
