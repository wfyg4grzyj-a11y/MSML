import pytest
from alpha_lab.market_data.sharadar_store import atomic_json
from alpha_lab.research.contracts import ResearchError,read_json
from alpha_lab.research.evidence import audit_evidence,attach_evidence


def test_evidence_audit_preserves_requirements_and_never_automatically_closes_cases(tmp_path):
    source=tmp_path/'queue.json';atomic_json(source,[{'id':'one','scope':{'security_id':'1','date':'2020-01-02'},
                              'status':'unresolved','required_evidence':['verified issuer mapping']}])
    summary=audit_evidence([source],None,tmp_path/'audit')
    assert summary['case_count']==2 and summary['resolved_by_this_audit']==0 and summary['investment_ready'] is False
    case=read_json(tmp_path/'audit/cases.json')[0]
    proof=tmp_path/'proof.txt';proof.write_text('A primary-source record for explicit human review')
    before=source.read_bytes()
    result=attach_evidence(tmp_path/'audit',case['case_id'],proof,tmp_path/'attachment')
    assert result['attestation_changed'] is False and result['status'].endswith('human_review')
    assert source.read_bytes()==before and read_json(tmp_path/'audit/cases.json')[0]['review_status']=='primary_evidence_required'
    (tmp_path/'audit/cases.json').write_text('[]')
    with pytest.raises(ResearchError):attach_evidence(tmp_path/'audit',case['case_id'],proof,tmp_path/'other')


def test_queue_version_overlap_and_bad_transition_inputs_are_rejected(tmp_path):
    source=tmp_path/'queue.json';atomic_json(source,[{'id':'one','scope':'all','status':'open','required_evidence':'record'}])
    with pytest.raises(ResearchError,match='Repeated'):audit_evidence([source,source],None,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    transitions=tmp_path/'transitions.json';atomic_json(transitions,{'cases':[{'ticker':'A'}]})
    with pytest.raises(ResearchError):audit_evidence([source],transitions,tmp_path/'bad')
