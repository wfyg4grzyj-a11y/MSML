"""Construction headroom stays causal and cannot weaken hard execution gates."""
import numpy as np
import pytest

from alpha_lab.research.contracts import ResearchError,parse
from alpha_lab.research.portfolio import PortfolioPolicy,PositionInput,exposures,project_portfolio
from alpha_lab.research.study_contracts import StudyPacket
from alpha_lab.research.study_engine import evaluate_study
from alpha_lab.research.study_inputs import pin_study
from alpha_lab.research.study_workflow import resolve_study
from alpha_lab.market_data.sharadar_store import atomic_json
from tests.test_risk_portfolio import policy,positions
from tests.test_integrated_study import packet


def targets():
    return dict(max_gross=.8,max_name=.2,max_turnover=1.8,net_tolerance=0.,beta_tolerance=0.)


def buffered_policy(**updates):
    data=policy(max_name=.25,net_tolerance=.01,beta_tolerance=.01).model_dump()
    data.update(target_limits=targets());data.update(updates)
    return parse(PortfolioPolicy,data)


@pytest.mark.parametrize('field,value',[('max_gross',1.01),('max_name',.251),('max_turnover',2.01),
                                     ('net_tolerance',.011),('beta_tolerance',.011)])
def test_target_limits_cannot_weaken_hard_limits(field,value):
    data=buffered_policy().model_dump();data['target_limits'][field]=value
    with pytest.raises(ResearchError):parse(PortfolioPolicy,data)


def test_absent_targets_preserve_legacy_policy_serialization():
    original=policy().model_dump()
    assert 'target_limits' not in original
    assert parse(PortfolioPolicy,original | {'target_limits':None}).model_dump()==original
    original_packet=packet()
    assert parse(StudyPacket,original_packet).model_dump()['portfolio_policy']==original_packet['portfolio_policy']


def test_independent_buffered_equal_weight_target_keeps_hard_caps():
    hard=buffered_policy();before=hard.model_dump()
    result=project_portfolio(positions(),['A','B','C','D'],np.eye(4)*.0001,1000.,hard)
    assert result['status']=='feasible'
    np.testing.assert_allclose(list(result['target_weights'].values()),[.2,.2,-.2,-.2],atol=1e-6)
    assert result['construction_limits']['max_name']==.2
    assert result['hard_limits']['max_name']==.25
    assert hard.model_dump()==before and result['hard_violations']==[]


def test_inactive_risk_union_does_not_apply_buffer_twice():
    items=positions()+[PositionInput(security_id='E',score=None,raw_price_usd=None,held_shares=0.,eligible=False,
                        tradable=False,borrow_available_shares=None,issuer_id=None,sector_id=None,beta=None)]
    result=project_portfolio(items,[p.security_id for p in items],np.eye(5)*.0001,1000.,buffered_policy())
    assert result['status']=='feasible'
    np.testing.assert_allclose(list(result['target_weights'].values()),[.2,.2,-.2,-.2,0.],atol=1e-6)


def test_retained_position_check_uses_hard_not_construction_caps():
    hard=buffered_policy();items=positions();weights=np.array([.21,.19,-.2,-.2])
    values,violations=exposures(items,weights,np.eye(4)*.0001,1000.,hard)
    assert values['name_max']>hard.target_limits.max_name
    assert values['name_max']<hard.max_name and not violations


def test_infeasible_frozen_buffer_never_falls_back_to_hard_target():
    items=positions();items[0]=items[0].model_copy(update={'held_shares':2.3,'tradable':False})
    result=project_portfolio(items,[p.security_id for p in items],np.eye(4)*.0001,1000.,buffered_policy())
    assert result['target_weights'] is None
    assert result['status']=='infeasible' and result['fallback']=='none'
    assert result['construction_limits']['max_name']==.2


def integrated_packet(buffered=True,price=102.):
    value=packet();value['portfolio_policy']=buffered_policy().model_dump()
    if not buffered:value['portfolio_policy'].pop('target_limits')
    for session in value['scenario']['sessions'][1:]:session['quotes']['A']['raw_close_usd']=price
    return value


def test_small_synthetic_gap_fills_buffered_targets_with_same_hard_policy(tmp_path):
    old=integrated_packet(False);new=integrated_packet()
    assert {k:v for k,v in new['portfolio_policy'].items() if k!='target_limits'}==old['portfolio_policy']
    first,_=evaluate_study(parse(StudyPacket,old),tmp_path)
    second,plans=evaluate_study(parse(StudyPacket,new),tmp_path)
    assert not any(o['filled_delta'] for o in first['orders'])
    assert sum(bool(o['filled_delta']) for o in second['orders'])==4
    assert plans['plans'][0]['hard_limits']['max_name']==.25
    assert all(not row.get('batch_violations') for row in second['orders'])


def test_large_gap_is_still_rejected_and_does_not_change_earlier_targets(tmp_path):
    normal=integrated_packet();changed=integrated_packet(price=300.)
    normal_ledger,normal_plans=evaluate_study(parse(StudyPacket,normal),tmp_path)
    changed_ledger,changed_plans=evaluate_study(parse(StudyPacket,changed),tmp_path)
    assert normal_plans['plans']==changed_plans['plans']
    assert any(o['filled_delta'] for o in normal_ledger['orders'])
    assert not any(o['filled_delta'] for o in changed_ledger['orders'])
    assert all(o['status']=='rejected_batch_constraints' for o in changed_ledger['orders'])


def test_buffers_are_part_of_bundle_and_computation_identity(tmp_path):
    specs=[]
    for i,value in enumerate((integrated_packet(False),integrated_packet())):
        path=tmp_path/f'packet-{i}.json';atomic_json(path,value)
        bundle=tmp_path/f'bundle-{i}';pin_study(path,bundle)
        specs.append(resolve_study(bundle,dict(family_id='headroom',hypothesis_id='reference',title='Synthetic headroom identity',
                 seed=42,threads=1,timeout_seconds=30,max_output_bytes=4000000,memory_limit_mb=1024)))
    assert specs[0].bundle_id!=specs[1].bundle_id
    assert specs[0].semantic_id()!=specs[1].semantic_id()
