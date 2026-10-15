"""Commands for the fixed US equity study API; the generic agent is optional."""
from pathlib import Path
import json

from .contracts import ResearchError, parse, read_json


def product_commands(commands):
    commands.add_parser('factor-platform-catalog',help='Public JKP153 catalog, provider reconstruction statuses and missing dependencies')
    prepare=commands.add_parser('factor-platform-prepare',help='Build an immutable pre-2023 large-cap universe/history bundle')
    prepare.add_argument('--sources',type=Path,required=True);prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--start',default='1999-03-31');prepare.add_argument('--end',default='2022-12-30')
    prepare.add_argument('--universe-policy',type=Path)
    resolve=commands.add_parser('factor-platform-spec')
    for name in ('inputs','fundamentals','policy','request','output'):resolve.add_argument('--'+name,type=Path,required=True)
    for name in ('factor-platform-register','factor-platform-run'):
        command=commands.add_parser(name)
        for flag in ('inputs','fundamentals','spec','registry'):command.add_argument('--'+flag,type=Path,required=True)
        if name.endswith('register'):
            command.add_argument('--family',type=Path,required=True);command.add_argument('--hypothesis',type=Path,required=True)
            command.add_argument('--parent')
        else:command.add_argument('--output',type=Path,required=True);command.add_argument('--retry-of')
    for name in ('factor-platform-inspect','factor-platform-reconcile'):
        command=commands.add_parser(name);command.add_argument('directory',type=Path)
        if name.endswith('reconcile'):command.add_argument('--registry',type=Path,required=True)
    pin_tools=commands.add_parser('risk-tools-pin',help='Pin accepted equity-risk tools inputs and the tested package identity')
    pin_tools.add_argument('--config',type=Path,required=True);pin_tools.add_argument('--output',type=Path,required=True)
    catalog=commands.add_parser('risk-tools-catalog',help='List exact dated covariance models in a pinned local library')
    catalog.add_argument('--library',type=Path,required=True)
    for name in ('risk-tools-query','risk-tools-project'):
        tool=commands.add_parser(name,help='Save local risk analytics or constrained research targets; no LLM or orders')
        tool.add_argument('--library',type=Path,required=True);tool.add_argument('--request',type=Path,required=True);tool.add_argument('--output',type=Path,required=True)
    inspect_tools=commands.add_parser('risk-tools-inspect',help='Verify a saved local risk operation without loading a model')
    inspect_tools.add_argument('directory',type=Path)
    pin=commands.add_parser('study-pin',help='Pin bounded raw-accounting inputs and explicit risk bindings')
    pin.add_argument('--packet',type=Path,required=True);pin.add_argument('--output',type=Path,required=True)
    pin.add_argument('--risk-sources',type=Path,help='JSON mapping binding IDs to already pinned risk directories')
    bridge=commands.add_parser('study-from-signal',help='Bridge a verified P2 score study into an explicit provisional raw-price cohort')
    for name in ('signal-run','development-bundle','template','output'):bridge.add_argument('--'+name,type=Path,required=True)
    spec=commands.add_parser('study-spec',help='Resolve explicit local study and resource controls')
    spec.add_argument('--bundle',type=Path,required=True);spec.add_argument('--request',type=Path,required=True);spec.add_argument('--output',type=Path,required=True)
    for name in ('study-check','study-register','study-run'):
        child=commands.add_parser(name)
        child.add_argument('--bundle',type=Path,required=True);child.add_argument('--spec',type=Path,required=True)
        if name!='study-check':child.add_argument('--registry',type=Path,required=True)
        if name=='study-register':
            child.add_argument('--family',type=Path,required=True);child.add_argument('--hypothesis',type=Path,required=True);child.add_argument('--parent')
        if name=='study-run':child.add_argument('--output',type=Path,required=True);child.add_argument('--retry-of')
    for name in ('study-inspect','study-reconcile'):
        child=commands.add_parser(name);child.add_argument('directory',type=Path)
        if name.endswith('reconcile'):child.add_argument('--registry',type=Path,required=True)
    risk=commands.add_parser('risk-pin',help='Consume official equity-risk Forecast.save files without modifying producer')
    risk.add_argument('--source',type=Path,required=True);risk.add_argument('--request',type=Path,required=True);risk.add_argument('--output',type=Path,required=True)
    check=commands.add_parser('risk-check');check.add_argument('--directory',type=Path,required=True);check.add_argument('--use',type=Path,required=True)
    for name in ('research-list','research-inspect','research-report','research-serve'):
        child=commands.add_parser(name);child.add_argument('--root',type=Path,required=True)
        if name in ('research-inspect','research-report'):child.add_argument('--run',required=True)
        if name=='research-serve':child.add_argument('--host',default='127.0.0.1');child.add_argument('--port',type=int,default=8766)
    design=commands.add_parser('public-design',help='Explicitly run bounded public proposal assistance; no local evaluation')
    design.add_argument('--config',type=Path,required=True);design.add_argument('--output',type=Path,required=True)
    evidence=commands.add_parser('evidence-audit',help='Preserve unresolved data and broker-evidence requirements')
    evidence.add_argument('--queue',type=Path,action='append',required=True);evidence.add_argument('--transitions',type=Path)
    evidence.add_argument('--output',type=Path,required=True)
    attach=commands.add_parser('evidence-attach',help='Attach primary evidence for review without changing attestations')
    attach.add_argument('--audit',type=Path,required=True);attach.add_argument('--case-id',required=True)
    attach.add_argument('--evidence',type=Path,required=True);attach.add_argument('--output',type=Path,required=True)


def product_action(args):
    if args.command.startswith('factor-platform-'):
        from .factor_workflow import PlatformSpec,resolve_platform,run_platform,inspect_platform,reconcile_platform,validate_spec
        from .factor_inputs import prepare_inputs,save_json
        if args.command=='factor-platform-catalog':
            from .jkp_signals import catalog
            return catalog()
        if args.command=='factor-platform-prepare':
            from .universes import UniversePolicy
            value=read_json(args.universe_policy) if args.universe_policy else None
            policy=parse(UniversePolicy,value.get('universe_policy',value)) if value else None
            return prepare_inputs(args.sources,args.output,start=args.start,end=args.end,policy=policy)
        if args.command=='factor-platform-spec':
            spec=resolve_platform(args.inputs,args.fundamentals,read_json(args.policy),read_json(args.request))
            save_json(args.output,spec.model_dump());return {'trial_id':spec.semantic_id(),'output':str(args.output)}
        if args.command=='factor-platform-inspect':return inspect_platform(args.directory)
        if args.command=='factor-platform-reconcile':return reconcile_platform(args.registry,args.directory)
        if args.command=='factor-platform-run':return run_platform(args.inputs,args.fundamentals,args.spec,args.output,args.registry,retry_of=args.retry_of)
        from .registry import Registry
        from .signal_contracts import Family,Hypothesis
        spec=parse(PlatformSpec,read_json(args.spec));validate_spec(spec,args.inputs,args.fundamentals)
        with Registry(args.registry) as registry:
            trial=registry.register(parse(Family,read_json(args.family)),parse(Hypothesis,read_json(args.hypothesis)),spec,parent=args.parent)
        return {'registered':trial}
    if args.command.startswith('risk-tools-'):
        from .risk_tools import PortfolioRiskService,inspect_risk_operation,pin_library,save_risk_operation
        if args.command=='risk-tools-pin':return pin_library(args.config,args.output)
        if args.command=='risk-tools-catalog':return PortfolioRiskService(args.library).catalog()
        if args.command=='risk-tools-inspect':return inspect_risk_operation(args.directory)
        return save_risk_operation(args.library,args.request,args.output,projection=args.command=='risk-tools-project')
    if args.command=='study-from-signal':
        from .signal_adapter import packet_from_signal
        return packet_from_signal(args.signal_run,args.development_bundle,args.template,args.output)
    if args.command=='evidence-audit':
        from .evidence import audit_evidence
        return audit_evidence(args.queue,args.transitions,args.output)
    if args.command=='evidence-attach':
        from .evidence import attach_evidence
        return attach_evidence(args.audit,args.case_id,args.evidence,args.output)
    from .service import ResearchService
    from .study_inputs import load_study,pin_study
    from .study_contracts import StudySpec
    from .study_workflow import inspect_study,reconcile_study,resolve_study,validate_study_spec
    if args.command=='study-pin':
        return pin_study(args.packet,args.output,read_json(args.risk_sources) if args.risk_sources else None).model_dump()
    if args.command=='study-spec':
        spec=resolve_study(args.bundle,read_json(args.request))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as target:json.dump(spec.model_dump(),target,indent=2,allow_nan=False)
        return {'trial_id':spec.semantic_id(),'output':str(args.output)}
    if args.command=='study-inspect':return inspect_study(args.directory)
    if args.command=='study-reconcile':return reconcile_study(args.registry,args.directory)
    if args.command=='study-run':
        return ResearchService(args.output.parent).run(args.bundle,args.spec,args.output,args.registry,retry_of=args.retry_of)
    if args.command.startswith('study-'):
        bundle,packet=load_study(args.bundle);spec=parse(StudySpec,read_json(args.spec));validate_study_spec(spec,bundle)
        if args.command=='study-check':
            from .study_engine import PortfolioController
            controller=PortfolioController(packet,args.bundle)
            for context in packet.contexts:
                session=next(s for s in packet.scenario.sessions if s.date==context.date)
                controller.risk(context,session)
            return {'ready_for_bounded_computation':True,'investment_ready':False,'llm_calls':0,'holdout_access':False,'trial_id':spec.semantic_id()}
        from .registry import Registry
        from .signal_contracts import Family,Hypothesis
        with Registry(args.registry) as registry:
            trial=registry.register(parse(Family,read_json(args.family)),parse(Hypothesis,read_json(args.hypothesis)),spec,args.parent)
        return {'trial_id':trial,'registered':True}
    if args.command=='risk-pin':
        from .risk_adapter import pin_forecast
        return pin_forecast(args.source,args.output,read_json(args.request)).model_dump()
    if args.command=='risk-check':
        from .risk_adapter import RiskUse,load_risk
        binding,covariance=load_risk(args.directory,parse(RiskUse,read_json(args.use)))
        return {'forecast_id':binding.forecast_id,'covered_securities':len(covariance),'valid_for_declared_use':True}
    if args.command.startswith('research-'):
        service=ResearchService(args.root)
        if args.command=='research-list':return service.list_runs()
        if args.command=='research-inspect':return service.inspect(args.run)
        if args.command=='research-report':return {'report':service.report(args.run)}
        from .web import serve
        serve(args.root,args.host,args.port)
        return {'viewer_stopped':True}
    if args.command=='public-design':
        from dotenv import load_dotenv
        from .public_design import run_public_design
        load_dotenv()
        return run_public_design(args.config,args.output)
    raise ResearchError('Unknown US equity research command')
