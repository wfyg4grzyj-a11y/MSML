"""Shared local API for CLI, notebooks and saved-research dashboard views."""
from __future__ import annotations

from pathlib import Path
import re

from .contracts import ResearchError, read_json


class ResearchService:
    def __init__(self, root):
        self.root=Path(root).resolve()

    def path(self, name):
        if not isinstance(name,str) or not name or len(name)>500:
            raise ResearchError('Choose a saved research identifier')
        parts=name.replace('\\','/').split('/')
        if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',p) or p in ('.','..') for p in parts):
            raise ResearchError('Research identifier is outside the configured library')
        path=self.root.joinpath(*parts)
        if (not path.is_dir() or not path.resolve().is_relative_to(self.root)
                or any(self.root.joinpath(*parts[:i]).is_symlink() for i in range(1,len(parts)+1))):
            raise ResearchError('Saved research is unavailable in this library')
        return path

    def inspect(self, name):
        path=self.path(name)
        record=read_json(path/'run.json')
        if record.get('schema_version')==4 and record.get('kind')=='portfolio_risk_tools':
            from .risk_tools import inspect_risk_operation
            return inspect_risk_operation(path)
        if record.get('schema_version')==3:
            from .study_workflow import inspect_study
            return inspect_study(path)
        if record.get('schema_version')==2:
            from .signal_workflow import inspect_signal
            return inspect_signal(path)
        if record.get('schema_version')==1 and 'scientific_disposition' in record:
            from .workflow import inspect_run
            return inspect_run(path)
        raise ResearchError('This artifact is not a supported saved research run')

    def progress(self, name):
        path=self.path(name)
        record=read_json(path/'run.json')
        if record.get('schema_version') not in (1,2,3,4) or 'scientific_disposition' not in record:
            raise ResearchError('Unsupported progress record')
        return {key:record.get(key) for key in ('status','stage','started_at','updated_at','finished_at',
                'failure','failure_code','peak_rss_mb','worker_pid','scientific_disposition','llm_calls')}

    def report(self, name):
        evidence=self.inspect(name)
        if not evidence.get('artifacts_verified'):raise ResearchError('Completed verified evidence is required before showing a report')
        target=self.path(name)/'report.txt'
        if target.stat().st_size>1_000_000:raise ResearchError('Research report exceeds the viewer byte limit')
        return target.read_text(encoding='utf-8')

    def list_runs(self, *, max_directories=2000, max_depth=3):
        if not self.root.is_dir():return {'runs':[],'truncated':False}
        pending=[(self.root,0)];runs=[];visited=0
        while pending and visited<max_directories:
            directory,depth=pending.pop();visited+=1
            if directory!=self.root and (directory/'run.json').is_file():
                name=directory.relative_to(self.root).as_posix()
                try:
                    record=self.progress(name)
                    runs.append({'id':name,**record})
                except (ResearchError,OSError,ValueError,TypeError):pass
            if depth>=max_depth:continue
            try:children=sorted(directory.iterdir(),reverse=True)
            except OSError:continue
            for child in children:
                if (child.is_dir() and not child.is_symlink() and not child.name.startswith(('.', 'pytest','test_'))
                        and child.name not in ('feature-cache','cache','locks','node_modules','worker-output')):
                    pending.append((child,depth+1))
        return {'runs':sorted(runs,key=lambda r:r.get('updated_at') or '',reverse=True),'truncated':bool(pending)}

    def run(self,bundle,spec,output,registry,*,retry_of=None):
        from .study_workflow import run_study
        return run_study(bundle,spec,output,registry,retry_of=retry_of)

    def reconcile(self,name,registry):
        from .study_workflow import reconcile_study
        return reconcile_study(registry,self.path(name))

    def compare_signals(self,names):
        from .signal_workflow import compare_signals
        return compare_signals([self.path(name) for name in names])

    def risk(self,library):
        """Explicit local analytics, independent of public-only LLM proposal tools."""
        from .risk_tools import PortfolioRiskService
        return PortfolioRiskService(library)
