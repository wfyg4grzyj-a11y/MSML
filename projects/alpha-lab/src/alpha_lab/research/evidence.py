"""Immutable evidence worklist; attaching records never manufactures an attestation."""
from pathlib import Path
import shutil

from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact, safe_file, verify_files
from .contracts import ResearchError, digest, read_json


def audit_evidence(queue_paths,transition_path,destination):
    destination=Path(destination).resolve()
    sources={};cases=[];seen=set()
    for index,path in enumerate(queue_paths):
        path=Path(path);rows=read_json(path)
        if not isinstance(rows,list):raise ResearchError('Evidence queue must be a list of dated/scoped work items')
        key='queue-'+str(index);sources[key]=artifact(path)
        for row in rows:
            if not isinstance(row,dict) or not {'id','scope','status','required_evidence'}<=row.keys():
                raise ResearchError('Evidence queue lacks case identity, scope or evidence requirements')
            identifier=str(row['id'])
            if identifier in seen:raise ResearchError('Repeated work item; select one explicit queue version')
            seen.add(identifier)
            cases.append({'case_id':digest({'source':sources[key],'id':identifier}), 'source':key,
                          'source_case_id':identifier,'kind':'historical_data_evidence','scope':row['scope'],
                          'prior_status':row['status'],'required_evidence':row['required_evidence'],
                          'source_record_sha256':digest(row),'review_status':'primary_evidence_required',
                          'blocks_investment_ready':True})
    if transition_path is not None:
        source=read_json(transition_path)
        if not isinstance(source,dict) or not isinstance(source.get('cases'),list):raise ResearchError('Unsupported transition clarification packet')
        sources['transitions']=artifact(transition_path)
        for row in source['cases']:
            if not isinstance(row,dict) or not {'ticker','date','attribute','required_resolution'}<=row.keys():
                raise ResearchError('Transition packet lacks dated identity and resolution requirements')
            cases.append({'case_id':digest({'source':sources['transitions'],'case':row}), 'source':'transitions',
                          'source_case_id':None,'kind':'dated_identity_clarification',
                          'scope':{k:row[k] for k in ('ticker','date','attribute')},
                          'prior_status':row.get('status'),'required_evidence':row['required_resolution'],
                          'source_record_sha256':digest(row),'review_status':'primary_evidence_required',
                          'blocks_investment_ready':True})
    cases.append({'case_id':digest({'global_requirement':'historical_broker_stock_loan'}),'source':'explicit_project_requirement',
                  'source_case_id':None,'kind':'stock_loan_evidence','scope':'historical short-sale feasibility and holding costs',
                  'prior_status':'unavailable','required_evidence':['dated broker locate/available shares','dated fee/rate terms and accrual basis','recall/expiry history'],
                  'source_record_sha256':None,'review_status':'primary_evidence_required','blocks_investment_ready':True})
    destination.mkdir(parents=True,exist_ok=False)
    atomic_json(destination/'cases.json',cases)
    summary={'kind':'local_evidence_audit_v1','sources':sources,'case_count':len(cases),
             'by_kind':{k:sum(c['kind']==k for c in cases) for k in sorted({c['kind'] for c in cases})},
             'resolved_by_this_audit':0,'investment_ready':False,'point_in_time_attested':False,
             'automatic_attestation':False,'provider_calls':0,'llm_calls':0}
    atomic_json(destination/'summary.json',summary)
    (destination/'report.txt').write_text('US equity external evidence worklist\n\n'
        'Original dated/scoped evidence requirements are preserved with source and record hashes.\n'
        'No case is closed by importing it. Attach primary broker/vendor/action evidence for explicit review.\n'
        'Short interest does not establish a locate, rate or recall history. Current corrected snapshots do not prove historical arrival.\n'
        'No numerical backtest, interface, model response or automatic flag upgrades these attestations.\n'
        f'Cases requiring primary evidence/review: {len(cases)}\nResolved by this audit: 0\n',encoding='utf-8')
    atomic_json(destination/'COMPLETE.json',{'kind':'evidence_audit_v1','files':{n:artifact(destination/n) for n in ('cases.json','summary.json','report.txt')}})
    return summary


def attach_evidence(audit,case_id,evidence_file,output):
    audit=Path(audit)
    marker=read_json(safe_file(audit,'COMPLETE.json'))
    if marker.get('kind')!='evidence_audit_v1' or set(marker.get('files',{}))!= {'cases.json','summary.json','report.txt'}:
        raise ResearchError('Unsupported evidence audit')
    verify_files(audit,marker['files'])
    cases=read_json(audit/'cases.json')
    if not any(c['case_id']==case_id for c in cases):raise ResearchError('Unknown evidence work item')
    source=Path(evidence_file)
    if source.is_symlink() or not source.is_file() or source.stat().st_size>25_000_000:
        raise ResearchError('Attach a bounded regular primary-evidence file')
    expected=artifact(source)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,output/'primary-evidence.bin')
    if artifact(output/'primary-evidence.bin')!=expected:raise ResearchError('Evidence changed during attachment')
    record={'case_id':case_id,'audit_manifest':artifact(audit/'COMPLETE.json'),'evidence':expected,
            'status':'attached_requires_explicit_human_review','attestation_changed':False}
    atomic_json(output/'attachment.json',record)
    atomic_json(output/'COMPLETE.json',{'kind':'evidence_attachment_v1','files':{n:artifact(output/n) for n in ('attachment.json','primary-evidence.bin')}})
    return record
