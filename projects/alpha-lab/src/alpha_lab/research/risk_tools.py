"""Local portfolio analytics over pinned equity-risk-tools 0.2.0 artifacts.

This is a consumer, not an estimator or a provider/agent connection. Historical
StudyPacket's stricter attested-risk contract remains separate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
import importlib
import json
import shutil
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, field_validator

from .contracts import Artifact, Hash, Identifier, ResearchError, StrictModel, digest, file_hash, iso_day, parse, read_json
from .portfolio import PortfolioPolicy, PositionInput, project_portfolio

MODELS = ('market_residual', 'fundamental_latent', 'direct_covariance')
MAX_BYTES = 2_000_000_000


def _dependency():
    try:
        module = importlib.import_module('equity_risk_tools')
        if version('prisma-equity-risk') != '0.2.0':
            raise ResearchError('Install the tested local prisma-equity-risk 0.2.0 wheel')
        return module
    except ImportError:
        raise ResearchError('Install the local equity-risk tools wheel; see docs/RISK_TOOLS.txt') from None


def _package_identity():
    _dependency()
    files = {}
    for name in ('equity_risk', 'equity_risk_tools'):
        root = Path(importlib.import_module(name).__file__).parent
        files.update({name+'/'+p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*.py'))})
    return {'version': version('prisma-equity-risk'), 'numpy': version('numpy'), 'source_sha256': files}


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _artifact(path):
    return {'sha256': file_hash(path), 'bytes': path.stat().st_size}


def _safe(root, name):
    parts = name.split('/')
    if (not parts or any(not p or p in ('.', '..') or '\\' in p or ':' in p for p in parts)
            or any((root.joinpath(*parts[:i]).is_symlink()
                    or getattr(root.joinpath(*parts[:i]), 'is_junction', lambda: False)()) for i in range(1, len(parts)+1))):
        raise ResearchError('Risk artifact path is outside its pinned library')
    target = root.joinpath(*parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ResearchError('Risk artifact path is outside its pinned library')
    return target


def _verify(root, files, extras):
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise ResearchError('Invalid risk artifact inventory')
    if sum(parse(Artifact, item).bytes for item in files.values()) > MAX_BYTES:
        raise ResearchError('Risk artifact byte limit exceeded')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | set(extras):
        raise ResearchError('Risk artifact inventory changed or is incomplete')
    for name, expected in files.items():
        target = _safe(root, name)
        if not target.is_file() or _artifact(target) != expected:
            raise ResearchError('Pinned risk artifact bytes changed')
    for name in extras:
        _safe(root, name)


def _identity(binding):
    return digest({key: value for key, value in binding.items() if key != 'library_id'})


def pin_library(config, output):
    """Copy only documented accepted-forecast inputs; never write to producer."""
    module = _dependency()
    config, output = Path(config).resolve(), Path(output).resolve()
    producer_sha = file_hash(config)
    api = module.RiskTools.from_config(config)
    catalog = api.catalog()  # Reject duplicates and incomplete numerical fits first.
    settings = read_json(config)
    if len(catalog['snapshots']) > 512:
        raise ResearchError('Pin at most 512 dated model snapshots per library')
    sources, paths = [], {}
    pinned_config = {'schema_version': 1, 'default_model': settings.get('default_model', 'market'),
                     'cache_size': settings.get('cache_size', 6), 'exports': [], 'research_runs': []}
    for kind in ('exports', 'research_runs'):
        for index, value in enumerate(settings.get(kind, [])):
            source = (config.parent/value).resolve()
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                raise ResearchError('Pin risk inputs to a separate new destination')
            prefix = f'{kind}/{index}'
            names = {'prepared-input-hashes.json'}
            if kind == 'exports':
                names |= {'manifest.json', 'acceptance.json', 'security-order.csv'}
                for model in MODELS:
                    if (source/model/'forecast.json').is_file():
                        names |= {f'{model}/forecast.json', f'{model}/forecast.npz'}
            else:
                names.add('metrics.json')
                for day in read_json(source/'metrics.json')['accepted_common_dates']:
                    iso_day(day)
                    for model in MODELS:
                        names |= {f'forecasts/{day}/{model}/forecast.json', f'forecasts/{day}/{model}/forecast.npz'}
            for name in sorted(names):
                target = _safe(source, name)
                if target.is_file():
                    paths[prefix+'/'+name] = target
            sources.append({'kind': kind, 'source': str(source)})
            pinned_config[kind].append(prefix)
    if len(paths) > 4095 or sum(p.stat().st_size for p in paths.values()) > MAX_BYTES:
        raise ResearchError('Risk library exceeds file or byte limit')
    before = {name: _artifact(path) for name, path in paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, source in paths.items():
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _write(output/'risk-tools.json', pinned_config)
    files = before | {'risk-tools.json': _artifact(output/'risk-tools.json')}
    binding = {'schema_version': 1, 'producer_config_sha256': producer_sha, 'sources': sources,
               'package': _package_identity(), 'files': files}
    binding['library_id'] = _identity(binding)
    # Validate copied matrices/mappings/acceptance eagerly, not just their headers.
    pinned_api = module.RiskTools.from_config(output/'risk-tools.json')
    for snapshot in pinned_api.catalog()['snapshots']:
        pinned_api.universe(as_of=snapshot['as_of'], model=snapshot['model'], limit=1)
    if producer_sha != file_hash(config) or before != {name: _artifact(path) for name, path in paths.items()}:
        raise ResearchError('Producer changed while pinning risk artifacts')
    _write(output/'risk-library.json', binding)
    _verify(output, files, {'risk-library.json'})
    _write(output/'COMPLETE.json', {'library_id': binding['library_id'], 'binding_sha256': file_hash(output/'risk-library.json')})
    return {'library_id': binding['library_id'], 'snapshots': len(catalog['snapshots']), 'output': str(output)}


def verify_library(directory, *, check_package=True):
    root = Path(directory).resolve()
    binding = read_json(root/'risk-library.json')
    complete = read_json(root/'COMPLETE.json')
    if (binding.get('schema_version') != 1 or binding.get('library_id') != _identity(binding)
            or complete != {'library_id': binding['library_id'], 'binding_sha256': file_hash(root/'risk-library.json')}):
        raise ResearchError('Incomplete or mismatched risk library')
    _verify(root, binding['files'], {'risk-library.json', 'COMPLETE.json'})
    if check_package and binding['package'] != _package_identity():
        raise ResearchError('Risk runtime differs from the pinned package; create a new library')
    return binding


class RiskProjection(StrictModel):
    as_of: str
    model: Literal['market', 'fundamental', 'direct', 'market_residual', 'fundamental_latent', 'direct_covariance']
    decision_time: str
    available_at: str
    availability_kind: Literal['declared_assumption', 'archived_observation']
    availability_evidence_sha256: Hash
    max_age_calendar_days: Annotated[int, Field(ge=0, le=30)]
    benchmark_id: Identifier
    cash_id: Identifier
    calendar_id: Identifier
    beta_evidence_sha256: Hash
    nav_usd: Annotated[float, Field(gt=0)]
    positions: Annotated[list[PositionInput], Field(min_length=2, max_length=512)]
    policy: PortfolioPolicy

    _date = field_validator('as_of')(iso_day)

    @field_validator('decision_time', 'available_at')
    @classmethod
    def timestamp(cls, value):
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            raise ValueError('Require an explicit UTC offset')
        return stamp.astimezone(timezone.utc).isoformat()


class PortfolioRiskService:
    """Same local service for notebooks/CLI; never attached to the public designer.

    Each operation verifies the source lock before AND after using the package's
    read-only cache. No arbitrary file paths are exposed in function arguments.
    """
    def __init__(self, library):
        self.root = Path(library).resolve()
        self.binding = verify_library(self.root)
        self._module = _dependency()
        self._executor = self._module.RiskToolExecutor.from_config(self.root/'risk-tools.json')

    def _check(self):
        binding = verify_library(self.root)
        if binding != self.binding:
            raise ResearchError('Risk library binding changed during this session')

    @property
    def schemas(self):
        return self._executor.schemas

    def invoke(self, name, arguments):
        self._check()
        result = self._executor.invoke(name, arguments)
        self._check()
        return result

    def __call__(self, name, arguments):
        return {'output': json.dumps(self.invoke(name, arguments), allow_nan=False)}

    def catalog(self):
        return self.invoke('risk_catalog', {})['result']

    def covariance(self, *, as_of, model, horizon, security_ids):
        """Exact caller order for local portfolio construction (Python, <=512)."""
        if not isinstance(security_ids, list) or not 1 <= len(security_ids) <= 512:
            raise ResearchError('Request 1 to 512 ordered permanent security IDs')
        self._check()
        try:
            result = self._executor.api.security_matrices(as_of=as_of, securities=security_ids,
                        model=model, horizon=horizon, identifier='security_id')
        except self._module.RiskToolError as error:
            raise ResearchError('Risk covariance unavailable: '+error.code) from None
        self._check()
        if result['security_ids'] != security_ids:
            raise ResearchError('Risk covariance order differs from requested permanent IDs')
        matrix = np.array(result['covariance'], dtype=np.float64)
        matrix.setflags(write=False)
        return matrix, result['metadata']

    def project(self, request):
        """Research targets only; retain AlphaLab's borrow/mandate constraints."""
        request = parse(RiskProjection, request)
        from ..equity_calendar import session_schedule
        # Forecast uses the full close; declaring midnight availability is invalid.
        day = datetime.fromisoformat(request.as_of).date()
        schedule, _ = session_schedule(str(day-timedelta(days=7)), str(day+timedelta(days=7)))
        sessions = schedule[schedule.date.dt.strftime('%Y-%m-%d') == request.as_of]
        if len(sessions) != 1:
            raise ResearchError('Risk as_of must be an official US equity session')
        close = datetime.fromisoformat(str(sessions.iloc[0].session_close_utc))
        available, decision = map(datetime.fromisoformat, (request.available_at, request.decision_time))
        age = (decision.date()-datetime.fromisoformat(request.as_of).date()).days
        if not close <= available <= decision or not 0 <= age <= request.max_age_calendar_days:
            raise ResearchError('Risk forecast is future, unavailable or stale at the decision')
        ids = [p.security_id for p in request.positions]
        covariance, metadata = self.covariance(as_of=request.as_of, model=request.model,
                        horizon=request.policy.risk_horizon_sessions, security_ids=ids)
        expected = {key: getattr(request, key) for key in ('benchmark_id', 'cash_id', 'calendar_id')}
        if metadata['data_bindings'] != expected:
            raise ResearchError('Risk benchmark, cash or calendar binding mismatch')
        target = project_portfolio(request.positions, ids, covariance, request.nav_usd, request.policy)
        return {'portfolio': target, 'risk': metadata, 'availability_kind': request.availability_kind,
                'availability_evidence_sha256': request.availability_evidence_sha256,
                'beta_evidence_sha256': request.beta_evidence_sha256, 'investment_ready': False,
                'scientific_disposition': 'provisional_local_risk_research_not_accepted',
                'orders_submitted': 0, 'historical_study_attestation': False}


def save_risk_operation(library, request, output, *, projection=False):
    """Persist successful or rejected local requests with a completion marker."""
    started = datetime.now(timezone.utc).isoformat()
    service = PortfolioRiskService(library)
    request = read_json(request) if isinstance(request, (str, Path)) else request
    if projection:
        response = {'ok': True, 'result': service.project(request)}
    else:
        if not isinstance(request, dict) or set(request) != {'tool', 'arguments'}:
            raise ResearchError('Risk request needs exactly tool and arguments')
        response = service.invoke(request['tool'], request['arguments'])
    service._check()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'request.json', request)
    _write(output/'response.json', response)
    _write(output/'risk-source.json', service.binding)
    now = datetime.now(timezone.utc).isoformat()
    from .runtime import environment_identity, implementation_identity
    record = {'schema_version': 4, 'kind': 'portfolio_risk_tools', 'status': 'completed' if response['ok'] else 'rejected',
              'stage': 'local_portfolio_projection' if projection else 'local_risk_query',
              'library_id': service.binding['library_id'], 'request_id': digest(request),
              'started_at': started, 'updated_at': now, 'finished_at': now,
              'scientific_disposition': 'provisional_local_risk_research_not_accepted',
              'investment_ready': False, 'llm_calls': 0, 'orders_submitted': 0,
              'implementation': implementation_identity(), 'environment': environment_identity() | {'scipy': version('scipy')},
              'package': service.binding['package']}
    _write(output/'run.json', record)
    report = ('AlphaLab local portfolio risk\nResearch only; no investment acceptance or orders.\n'
              'Exact saved date; annualized equivalent is a rate, not a one-year forecast.\n'
              'Equity risk excludes financing, borrow, execution and cash-rate risk.\n\n'
              +json.dumps(response, indent=2, allow_nan=False)+'\n')
    (output/'report.txt').write_text(report, encoding='utf-8')
    names = {'request.json', 'response.json', 'risk-source.json', 'run.json', 'report.txt'}
    _write(output/'COMPLETE.json', {'schema_version': 4, 'library_id': record['library_id'],
                                 'files': {name: _artifact(output/name) for name in sorted(names)}})
    return inspect_risk_operation(output)


def inspect_risk_operation(directory):
    """Verify an archive without requiring current estimator code or the source."""
    root = Path(directory)
    complete = read_json(root/'COMPLETE.json')
    names = {'request.json', 'response.json', 'risk-source.json', 'run.json', 'report.txt'}
    if complete.get('schema_version') != 4 or set(complete.get('files', {})) != names:
        raise ResearchError('Incomplete local risk operation')
    _verify(root, complete['files'], {'COMPLETE.json'})
    record, source, request, response = (read_json(root/name) for name in ('run.json', 'risk-source.json', 'request.json', 'response.json'))
    if (record.get('schema_version') != 4 or record.get('kind') != 'portfolio_risk_tools'
            or record.get('request_id') != digest(request) or source.get('library_id') != _identity(source)
            or source['library_id'] != record.get('library_id') or source['library_id'] != complete.get('library_id')
            or record.get('status') != ('completed' if response.get('ok') is True else 'rejected')
            or record.get('investment_ready') is not False or record.get('llm_calls') != 0):
        raise ResearchError('Local risk archive semantic binding mismatch')
    return {'artifacts_verified': True, 'status': record['status'], 'stage': record['stage'],
            'library_id': source['library_id'], 'request_id': record['request_id'], 'investment_ready': False,
            'llm_calls': 0, 'orders_submitted': 0}
