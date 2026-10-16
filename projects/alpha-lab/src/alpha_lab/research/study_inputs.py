"""Small immutable job-facing input packet; never follow full-history source paths."""
from pathlib import Path
import shutil

from alpha_lab.market_data.sharadar_store import atomic_json
from .bundles import artifact
from .contracts import ResearchError, digest, parse, read_json
from .ledger import validate_calendar
from .study_contracts import StudyBundle, StudyPacket


def verify_tree(directory, files):
    directory=Path(directory).resolve()
    for name, expected in files.items():
        parts=Path(name).parts
        target=directory/name
        if (not parts or any(p in ('..','.') for p in parts) or Path(name).is_absolute()
                or not target.resolve().is_relative_to(directory) or any((directory/Path(*parts[:i])).is_symlink() for i in range(1,len(parts)+1))
                or not target.is_file() or artifact(target) != (expected.model_dump() if hasattr(expected,'model_dump') else expected)):
            raise ResearchError('Study input artifact is absent, changed or outside its declared bundle')


def pin_study(packet_path, destination, risk_sources=None):
    if Path(packet_path).stat().st_size>64_000_000:raise ResearchError('Study packet exceeds the 64 MB reference input cap')
    packet=parse(StudyPacket,read_json(packet_path))
    validate_calendar(packet.scenario)
    risk_sources=risk_sources or {}
    required={c.risk_binding_id for c in packet.contexts if c.risk_binding_id}
    if set(risk_sources)!=required:
        raise ResearchError('Provide exactly the pinned risk bindings referenced by the study')
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'INCOMPLETE').write_text('Require COMPLETE.json; no overwrite or automatic resume\n')
    atomic_json(destination/'packet.json',packet.model_dump())
    files={'packet.json':artifact(destination/'packet.json')}
    for key,source in risk_sources.items():
        target=destination/key;target.mkdir()
        source=Path(source).resolve()
        for name in ('forecast.json','forecast.npz','risk-binding.json','COMPLETE.json'):
            if not (source/name).is_file() or (source/name).is_symlink():raise ResearchError('Incomplete source risk binding')
            original=artifact(source/name)
            shutil.copyfile(source/name,target/name)
            if artifact(target/name)!=original:raise ResearchError('Risk binding changed while copying')
            files[key+'/'+name]=original
    body={'schema_version':3,'start':packet.scenario.sessions[0].date,'end':packet.scenario.sessions[-1].date,
          'packet_sha256':files['packet.json']['sha256'],'risk_ids':sorted(required),'files':files}
    bundle=parse(StudyBundle,body | {'bundle_id':digest(body)})
    atomic_json(destination/'bundle.json',bundle.model_dump())
    atomic_json(destination/'COMPLETE.json',{'kind':'raw_ledger_bundle_v3','bundle_id':bundle.bundle_id,
                                           'files':files | {'bundle.json':artifact(destination/'bundle.json')}})
    load_study(destination)
    (destination/'INCOMPLETE').unlink()
    return bundle


def load_study(directory):
    directory=Path(directory).resolve()
    marker=read_json(directory/'COMPLETE.json')
    if set(marker)!= {'kind','bundle_id','files'} or marker['kind']!='raw_ledger_bundle_v3':
        raise ResearchError('Invalid study bundle completion')
    verify_tree(directory,marker['files'])
    bundle=parse(StudyBundle,read_json(directory/'bundle.json'))
    if bundle.identity()!=bundle.bundle_id or marker['bundle_id']!=bundle.bundle_id or marker['files']!=({k:v.model_dump() for k,v in bundle.files.items()} | {'bundle.json':artifact(directory/'bundle.json')}):
        raise ResearchError('Study bundle identity/inventory mismatch')
    if (directory/'packet.json').stat().st_size>64_000_000:raise ResearchError('Study packet exceeds the 64 MB reference input cap')
    packet=parse(StudyPacket,read_json(directory/'packet.json'))
    if (artifact(directory/'packet.json')['sha256']!=bundle.packet_sha256
            or [bundle.start,bundle.end]!=[packet.scenario.sessions[0].date,packet.scenario.sessions[-1].date]):
        raise ResearchError('Study packet bounds/hash mismatch')
    required={c.risk_binding_id for c in packet.contexts if c.risk_binding_id}
    expected={'packet.json'} | {key+'/'+n for key in required for n in ('forecast.json','forecast.npz','risk-binding.json','COMPLETE.json')}
    if set(bundle.files)!=expected or bundle.risk_ids!=sorted(required):raise ResearchError('Unexpected job-facing study dependencies')
    validate_calendar(packet.scenario)
    return bundle,packet
