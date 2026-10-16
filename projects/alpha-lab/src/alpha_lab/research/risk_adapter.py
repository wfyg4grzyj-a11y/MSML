"""Read-only, versioned consumer of equity-risk Forecast.save artifacts."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal
import zipfile

import numpy as np
from pydantic import Field, field_validator

from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact, safe_file, verify_files
from .contracts import Artifact, Hash, Identifier, ResearchError, StrictModel, digest, iso_day, parse, read_json
from .ledger_contracts import utc_stamp


class RiskBinding(StrictModel):
    schema_version: Literal[1]
    forecast_id: Hash
    contract_sha256: Hash
    source_policy_id: Identifier
    as_of: str
    available_at: str
    training_cutoff: str
    availability_kind: Literal['archived_observation', 'declared_assumption', 'synthetic_reference']
    availability_evidence_sha256: Hash
    benchmark_id: Identifier
    cash_id: Identifier
    calendar_id: Identifier
    identity_namespace: Literal['sharadar_permaticker_share_class', 'synthetic_share_class']
    history_mode: Literal['synthetic_reference', 'public_information_reconstruction', 'archived_vendor_feed']
    point_in_time_attested: bool
    terminal_audit_complete: bool
    units: Literal['squared_decimal_return']
    target: Literal['additive_excess_returns']
    security_ids: Annotated[list[Identifier], Field(min_length=1, max_length=5000)]
    horizons: Annotated[list[int], Field(min_length=1, max_length=63)]
    files: dict[str, Artifact]

    _dates = field_validator('as_of', 'training_cutoff')(iso_day)
    _time = field_validator('available_at')(utc_stamp)

    def identity(self):
        return digest(self.model_dump(exclude={'forecast_id'}))


class RiskUse(StrictModel):
    decision_time: str
    horizon_sessions: Annotated[int, Field(ge=1, le=63)]
    max_age_calendar_days: Annotated[int, Field(ge=0, le=30)]
    contract_sha256: Hash
    source_policy_id: Identifier
    benchmark_id: Identifier
    cash_id: Identifier
    calendar_id: Identifier
    identity_namespace: Literal['sharadar_permaticker_share_class', 'synthetic_share_class']
    expected_security_order: Annotated[list[Identifier], Field(min_length=1, max_length=5000)]
    use: Literal['synthetic_validation', 'attested_research']

    _time = field_validator('decision_time')(utc_stamp)


def read_forecast(directory):
    directory = Path(directory)
    meta = read_json(safe_file(directory, 'forecast.json'))
    if (set(meta) != {'model', 'as_of', 'horizons', 'units', 'target', 'diagnostics'}
            or meta['units'] != 'squared_decimal_return' or meta['target'] != 'additive_excess_returns'
            or not isinstance(meta['horizons'], list) or not meta['horizons']
            or any(type(h) is not int or not 1 <= h <= 63 for h in meta['horizons'])
            or len(set(meta['horizons'])) != len(meta['horizons'])):
        raise ResearchError('Unsupported risk forecast units, target or schema')
    with zipfile.ZipFile(safe_file(directory, 'forecast.npz')) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 512_000_000:
            raise ResearchError('Risk forecast exceeds the uncompressed array cap')
    with np.load(directory/'forecast.npz', allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    ids = arrays.pop('security_ids', None)
    if (ids is None or ids.ndim != 1 or ids.dtype.kind != 'U' or not 1 <= len(ids) <= 5000
            or len(set(ids.tolist())) != len(ids)):
        raise ResearchError('Risk coverage requires unique stable Unicode share-class IDs')
    expected = {'covariance_'+str(h) for h in meta['horizons']}
    if not expected <= arrays.keys():
        raise ResearchError('Risk horizon arrays are incomplete')
    for key, value in arrays.items():
        if key not in expected and not any(key.startswith('component_'+str(h)+'_') for h in meta['horizons']):
            raise ResearchError('Unrecognized risk array')
        if value.shape != (len(ids), len(ids)) or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ResearchError('Risk covariance/component dimensions or values are invalid')
        if not np.allclose(value, value.T, atol=1e-12, rtol=0):
            raise ResearchError('Risk covariance/component is asymmetric')
        if key in expected and np.linalg.eigvalsh(value).min() < -1e-12:
            raise ResearchError('Risk covariance is not positive semidefinite')
    return meta, ids.tolist(), arrays


def pin_forecast(source, destination, request):
    """Copy only official forecast files; never modify/import the sibling package."""
    import shutil
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ResearchError('Pin forecast outside its source artifact directory')
    hashes = {name: artifact(safe_file(source, name)) for name in ('forecast.json', 'forecast.npz')}
    meta, ids, _ = read_forecast(source)
    reserved = {'schema_version', 'forecast_id', 'as_of', 'units', 'target', 'security_ids', 'horizons', 'files'}
    if set(request) & reserved:
        raise ResearchError('Risk binding request contains reserved producer fields')
    body = request | {'schema_version': 1, 'as_of': meta['as_of'], 'units': meta['units'], 'target': meta['target'],
                      'security_ids': ids, 'horizons': meta['horizons'], 'files': hashes}
    binding = parse(RiskBinding, body | {'forecast_id': digest(body)})
    validate_binding(binding, meta)
    verify_files(source, hashes)
    destination.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        shutil.copyfile(source/name, destination/name)
    atomic_json(destination/'risk-binding.json', binding.model_dump())
    verify_files(destination, hashes)
    atomic_json(destination/'COMPLETE.json', {'kind': 'risk_binding_v1', 'forecast_id': binding.forecast_id,
                'files': {name: artifact(destination/name) for name in (*hashes, 'risk-binding.json')}})
    return binding


def validate_binding(binding, meta):
    if binding.identity() != binding.forecast_id or set(binding.files) != {'forecast.json', 'forecast.npz'}:
        raise ResearchError('Risk binding identity/inventory mismatch')
    if binding.training_cutoff > binding.as_of or binding.as_of > '2022-12-31':
        raise ResearchError('Risk forecast/calibration exceeds development information bounds')
    if datetime.fromisoformat(binding.available_at).date().isoformat() < binding.as_of:
        raise ResearchError('Forecast cannot be available before its observation date')
    from alpha_lab.equity_calendar import session_schedule
    day=datetime.fromisoformat(binding.as_of)
    calendar, _ = session_schedule((day-timedelta(days=7)).date().isoformat(), (day+timedelta(days=1)).date().isoformat())
    matching=[row for row in calendar.itertuples() if str(row.date.date())==binding.as_of]
    if len(matching) != 1 or datetime.fromisoformat(binding.available_at) < datetime.fromisoformat(str(matching[0].session_close_utc)):
        raise ResearchError('Close-based risk forecast cannot precede the actual as-of session close')
    diag = meta.get('diagnostics', {})
    if binding.history_mode != 'synthetic_reference':
        provenance = diag.get('data_provenance', {})
        if (not binding.point_in_time_attested or not binding.terminal_audit_complete
                or provenance.get('point_in_time') is not True or provenance.get('terminal_actions_reconciled') is not True
                or provenance.get('cap_basis') != 'total_issuer' or provenance.get('industry_scheme') != 'SIC'
                or provenance.get('return_convention') != 'simple_total'
                or binding.availability_kind != 'archived_observation'):
            raise ResearchError('Real risk artifacts lack the required producer/availability evidence')
        for key in ('benchmark_id', 'cash_id', 'calendar_id'):
            if provenance.get(key) != getattr(binding, key):
                raise ResearchError('Risk producer and binding data conventions differ')
        cutoffs = [diag.get('prior_calibration', {}).get('calibration_end'), diag.get('parameter_selection_end')]
        known = [v for v in cutoffs if isinstance(v, str)]
        if not known or max(known) != binding.training_cutoff:
            raise ResearchError('Risk training/selection cutoff lacks matching producer evidence')


def load_risk(directory, use):
    directory = Path(directory)
    marker = read_json(safe_file(directory, 'COMPLETE.json'))
    if set(marker) != {'kind', 'forecast_id', 'files'} or marker['kind'] != 'risk_binding_v1' or set(marker['files']) != {'forecast.json', 'forecast.npz', 'risk-binding.json'}:
        raise ResearchError('Incomplete risk artifact publication')
    verify_files(directory, marker['files'])
    binding = parse(RiskBinding, read_json(directory/'risk-binding.json'))
    if marker['forecast_id'] != binding.forecast_id:
        raise ResearchError('Risk completion identity mismatch')
    verify_files(directory, binding.files)
    meta, ids, arrays = read_forecast(directory)
    validate_binding(binding, meta)
    if ids != binding.security_ids or meta['as_of'] != binding.as_of or meta['horizons'] != binding.horizons:
        raise ResearchError('Risk producer identity differs from binding')
    validate_risk_use(binding,use)
    covariance = arrays['covariance_'+str(use.horizon_sessions)].astype('float64', copy=True)
    covariance.setflags(write=False)
    return binding, covariance


def validate_risk_use(binding,use):
    """Repeat time/policy checks even when immutable validated arrays are reused."""
    for key in ('contract_sha256', 'source_policy_id', 'benchmark_id', 'cash_id', 'calendar_id', 'identity_namespace'):
        if getattr(binding,key) != getattr(use,key):
            raise ResearchError('Risk adapter contract/policy/convention mismatch')
    decision = datetime.fromisoformat(use.decision_time)
    if (datetime.fromisoformat(binding.available_at) > decision or binding.as_of > decision.date().isoformat()
            or (decision.date()-datetime.fromisoformat(binding.as_of).date()).days > use.max_age_calendar_days
            or binding.training_cutoff > decision.date().isoformat()):
        raise ResearchError('Risk forecast is future, stale or trained beyond the decision cutoff')
    if binding.history_mode == 'synthetic_reference' and use.use != 'synthetic_validation':
        raise ResearchError('Synthetic forecasts cannot be promoted to attested research')
    if use.expected_security_order != binding.security_ids or use.horizon_sessions not in binding.horizons:
        raise ResearchError('Risk security ordering/held coverage or horizon mismatch; no implicit reordering/scaling')


def diagonal_baseline(ids, variance_per_session, horizon_sessions):
    if (len(set(ids)) != len(ids) or not ids or not np.isfinite(variance_per_session) or variance_per_session <= 0
            or horizon_sessions != 1):
        raise ResearchError('Diagonal baseline requires explicit positive one-session decimal variance')
    return np.eye(len(ids), dtype='float64')*variance_per_session
