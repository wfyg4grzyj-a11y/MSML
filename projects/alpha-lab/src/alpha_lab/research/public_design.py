"""Bounded public recipe assistance; no market/result, evaluator or attestation tools."""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
from urllib.parse import urlsplit

from alpha_lab.market_data.sharadar_store import atomic_json
from .contracts import ResearchError, digest, file_hash, parse, read_json
from .features import feature_definition
from .signal_contracts import FEATURES, Recipe, baseline_recipes


def tool(name,description,properties,required):
    return {'type':'function','name':name,'description':description,'strict':True,
            'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}


DESIGN_TOOLS=[tool('research_catalog','Read public feature definitions and fixed recipe operators',{},[]),
              tool('draft_recipe','Stage a public linear recipe for explicit later registration; no evaluation',
                   {'name':{'type':'string'},'factors':{'type':'array','items':{'type':'object','properties':{
                       'feature':{'type':'string','enum':list(FEATURES)},'coefficient':{'type':'number'}},
                       'required':['feature','coefficient'],'additionalProperties':False}},'premise':{'type':'string'}},
                   ['name','factors','premise']),
              tool('validate_draft','Check a previously staged public recipe against supported operators',
                   {'draft_id':{'type':'string'}},['draft_id']),
              tool('report_to_user','Finish public design with assumptions and proposed tests; metrics unavailable',
                   {'summary':{'type':'string'}},['summary'])]

PUBLIC_PROMPT=('Design a public US equity fundamental/market-data market-neutral hypothesis. Use only this fixed tool catalog. '
               'Stage a small supported recipe, validate it, then report the economic premise and proposed tests. '
               'No actual market data, local coverage, results, files, shell, evaluator generation, budget edits, holdout reveal '
               'or data attestations are available. Do not invent performance. Local registration and testing are separate explicit actions.')


class PublicSignalTools:
    """Only this new public output directory is writable through these tools.

    This narrows application capabilities; it does not establish an OS sandbox.
    """
    def __init__(self,directory,max_drafts):
        self.directory=Path(directory)
        self.max_drafts=max_drafts
        self.drafts={};self.report=None

    def __call__(self,name,args):
        if not isinstance(args,dict):raise ResearchError('Public design requires structured arguments')
        allowed={'research_catalog':set(),'draft_recipe':{'name','factors','premise'},'validate_draft':{'draft_id'},'report_to_user':{'summary'}}
        if name not in allowed or set(args)!=allowed[name]:raise ResearchError('Tool/fields are outside public design scope')
        if name=='research_catalog':
            out={'features':[feature_definition(n) for n in FEATURES],'fixed_recipes':baseline_recipes(),
                 'data_scope':'public definitions only; no datasets, coverage counts, results or arbitrary code',
                 'testing':'Explicit human/local registration, folds and budgets required before evaluation',
                 'permissions':'Cannot modify core evaluator, limits, split authority or attestations'}
        elif name=='draft_recipe':
            if not isinstance(args['factors'],list) or not 1<=len(args['factors'])<=len(FEATURES):raise ResearchError('Invalid public factor list')
            if any(not isinstance(f,dict) or set(f)!= {'feature','coefficient'} for f in args['factors']):raise ResearchError('Invalid public factor fields')
            factors={f['feature']:f['coefficient'] for f in args['factors']}
            if len(factors)!=len(args['factors']):raise ResearchError('Duplicate public factors')
            recipe=parse(Recipe,{'name':args['name'],'operator':'linear','factors':factors,'transform':'cross_sectional_rank'})
            if not isinstance(args['premise'],str) or not 10<=len(args['premise'])<=2000:raise ResearchError('A bounded economic premise is required')
            identity=digest(recipe.computation())
            if identity not in self.drafts:
                if len(self.drafts)>=self.max_drafts:raise ResearchError('Public proposal budget exhausted')
                self.drafts[identity]={'draft_id':identity,'recipe':recipe.model_dump(),'premise':args['premise'],
                                      'status':'unregistered_unevaluated_public_proposal'}
                atomic_json(self.directory/'drafts.json',list(self.drafts.values()))
            out={'draft_id':identity,'status':'staged_for_explicit_local_review','performance':'unknown'}
        elif name=='validate_draft':
            if args['draft_id'] not in self.drafts:raise ResearchError('Unknown public draft')
            parse(Recipe,self.drafts[args['draft_id']]['recipe'])
            out={'valid_supported_recipe':True,'implementation':'fixed trusted signal engine; no generated evaluator'}
        else:
            if not self.drafts or not isinstance(args['summary'],str) or not 1<=len(args['summary'])<=16000:
                raise ResearchError('Finish after staging a bounded proposal with a public summary')
            self.report=args['summary']
            (self.directory/'public-report.txt').write_text(self.report,encoding='utf-8')
            return {'output':'Public design saved; local evaluation not started','done':True}
        return {'output':json.dumps(out,allow_nan=False)}


def run_public_design(config_path,destination):
    """The sole network entry is an explicit call with all reviewed LLM controls."""
    from alpha_lab.llm_runtime import LLMSettings,bounded_provider
    config=read_json(config_path)
    if set(config)!= {'data_scope','base_url','llm','max_drafts'} or config['data_scope']!='public_only':
        raise ResearchError('Public design requires exact scope, endpoint, controls and proposal cap')
    if not isinstance(config['llm'],dict) or set(config['llm'])!= {f.name for f in fields(LLMSettings)}:
        raise ResearchError('Intentionally set every LLM control; no run defaults')
    settings=LLMSettings(**config['llm'])
    if settings.deadline_utc is None or datetime.fromisoformat(settings.deadline_utc)<=datetime.now(timezone.utc):
        raise ResearchError('Public design needs a future absolute authorization deadline')
    if type(config['max_drafts']) is not int or not 1<=config['max_drafts']<=4:raise ResearchError('Choose one to four public proposals')
    endpoint=urlsplit(config['base_url'])
    if endpoint.scheme!='https' or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ResearchError('Public design endpoint must be HTTPS without credentials or query')
    if os.environ.get('OPENAI_BASE_URL','').rstrip('/')!=config['base_url'].rstrip('/') or not os.environ.get('OPENAI_API_KEY','').strip():
        raise ResearchError('Configure local endpoint/key before explicitly starting public design')
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    from .runtime import implementation_identity,environment_identity
    package=Path(__file__).resolve().parents[1]
    state={'kind':'public_signal_design_v4','status':'running','data_scope':'public_only','llm':asdict(settings),'base_url':config['base_url'],
           'started_at':datetime.now(timezone.utc).isoformat(),'finished_at':None,
           'prompt_sha256':digest(PUBLIC_PROMPT),'tool_schema_sha256':digest(DESIGN_TOOLS),
           'implementation':implementation_identity(),'environment':environment_identity(),
           'agent_runtime':{str(p.relative_to(package)).replace('\\','/'):file_hash(p)
                            for p in sorted(package.glob('*.py'))+sorted((package/'providers').glob('*.py'))},
           'summary_policy':'same bounded provider/model/reasoning/ledger; public context only',
           'generation_seed':'not set; model generation is not claimed deterministic',
           'tools':[t['name'] for t in DESIGN_TOOLS],'max_drafts':config['max_drafts'],'model_context':'fixed public definitions and model proposals only',
           'sdk_retries':0,'automatic_retries':0,'concurrency':1,'store':False,'temperature':'omitted','top_p':'omitted',
           'retention':'Gateway policy unknown; store=false is not a retention guarantee','scientific_disposition':'unevaluated_public_proposals',
           'resume':'none; preserve permanent usage intent after failure','local_evaluation_started':False}
    atomic_json(destination/'design-run.json',state)
    (destination/'public-system.txt').write_text(PUBLIC_PROMPT,encoding='utf-8')
    atomic_json(destination/'public-tools.json',DESIGN_TOOLS)
    provider=None
    try:
        from alpha_lab.agent import AgentLoop
        from alpha_lab.context import ContextManager
        provider=bounded_provider(settings,destination/'llm-usage.json')
        tools=PublicSignalTools(destination,config['max_drafts'])
        agent=AgentLoop(provider,settings.model,ContextManager(provider,settings.model,str(destination)),lambda event:None,
                        reasoning_effort=settings.reasoning_effort,tools=DESIGN_TOOLS,tool_executor=tools,
                        prompt_builder=lambda *args:PUBLIC_PROMPT,min_report_attempts=1)
        agent.run('Draft and validate a bounded public US equity signal hypothesis.')
        if not agent._done or tools.report is None:raise ResearchError('Public design did not complete its required report')
        state.update(status='completed',draft_count=len(tools.drafts),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException:
        state.update(status='failed',failure='public_design_failed_no_automatic_retry',finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(destination/'design-run.json',state)
        raise ResearchError('Public design failed; preserve its usage intent before any new attempt') from None
    finally:
        pending_error=sys.exc_info()[0] is not None
        try:
            if provider is not None and callable(getattr(provider,'close',None)):provider.close()
        except Exception:
            state.update(status='failed',failure='provider_cleanup_failed_no_automatic_retry',finished_at=datetime.now(timezone.utc).isoformat())
            atomic_json(destination/'design-run.json',state)
            if not pending_error:raise ResearchError('Public design provider cleanup failed; preserve its usage intent') from None
    atomic_json(destination/'design-run.json',state)
    return state
