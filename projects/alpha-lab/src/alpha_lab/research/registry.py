"""Local SQLite trial and knowledge ledger; transactions never include file publication."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid

from .contracts import ResearchError, canonical, digest


def stamp():
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE families(id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE hypotheses(id TEXT PRIMARY KEY, family TEXT NOT NULL REFERENCES families(id), body TEXT NOT NULL);
CREATE TABLE trials(id TEXT PRIMARY KEY, body TEXT NOT NULL, comparison_id TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL, created TEXT NOT NULL);
CREATE TABLE registrations(hypothesis TEXT NOT NULL REFERENCES hypotheses(id),
                           trial TEXT NOT NULL REFERENCES trials(id), parent TEXT REFERENCES trials(id),
                           created TEXT NOT NULL, PRIMARY KEY(hypothesis,trial));
CREATE TABLE attempts(id TEXT PRIMARY KEY, trial TEXT NOT NULL REFERENCES trials(id),
                      hypothesis TEXT NOT NULL REFERENCES hypotheses(id), path TEXT NOT NULL UNIQUE,
                      status TEXT NOT NULL CHECK(status IN ('running','failed','completed')),
                      retry_of TEXT REFERENCES attempts(id), created TEXT NOT NULL, finished TEXT,
                      result_hash TEXT, failure TEXT);
CREATE TABLE exposures(attempt TEXT PRIMARY KEY REFERENCES attempts(id), trial TEXT NOT NULL REFERENCES trials(id),
                       hypothesis TEXT NOT NULL REFERENCES hypotheses(id), period_start TEXT NOT NULL,
                       period_end TEXT NOT NULL, result_hash TEXT, created TEXT NOT NULL,
                       kind TEXT NOT NULL);
CREATE TABLE events(sequence INTEGER PRIMARY KEY, at TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE prior_exposures(id TEXT PRIMARY KEY, body TEXT NOT NULL, created TEXT NOT NULL);
PRAGMA user_version=2;
"""


def initialize_registry(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb'):
        pass
    db = sqlite3.connect(path)
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript(SCHEMA)
        db.commit()
    finally:
        db.close()


class Registry:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path).resolve()
        if not self.path.is_file() or self.path.is_symlink():
            raise ResearchError('Initialize an explicit research registry first')
        self.db = sqlite3.connect(self.path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'),
                                  uri=True, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        if readonly:
            self.db.execute('PRAGMA query_only=ON')
        if self.db.execute('PRAGMA user_version').fetchone()[0] != 2:
            self.db.close()
            raise ResearchError('Unsupported registry schema; do not overwrite or migrate implicitly')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def event(self, kind, body):
        self.db.execute('INSERT INTO events(at,kind,body) VALUES(?,?,?)', (stamp(), kind, canonical(body)))

    def immutable(self, table, key, body, *extra):
        found = self.db.execute(f'SELECT body FROM {table} WHERE id=?', (key,)).fetchone()
        if found:
            if found['body'] != canonical(body):
                raise ResearchError('Registered family/hypothesis is immutable; use a linked new registration')
        else:
            values = (key, *extra, canonical(body))
            self.db.execute(f'INSERT INTO {table} VALUES('+','.join('?' for _ in values)+')', values)

    def register(self, family, hypothesis, spec, parent=None):
        if family.family_id != hypothesis.family_id or (spec.family_id, spec.hypothesis_id) != (family.family_id, hypothesis.hypothesis_id):
            raise ResearchError('Family, hypothesis and resolved study do not match')
        trial = spec.semantic_id()
        with self.transaction():
            self.immutable('families', family.family_id, family.model_dump())
            self.immutable('hypotheses', hypothesis.hypothesis_id, hypothesis.model_dump(), family.family_id)
            registered = self.db.execute('SELECT parent FROM registrations WHERE hypothesis=? AND trial=?',
                                         (hypothesis.hypothesis_id, trial)).fetchone()
            if registered and registered['parent'] != parent:
                raise ResearchError('Registered parent lineage cannot change')
            if registered:
                return trial
            if parent == trial or (parent and not self.db.execute('SELECT 1 FROM trials WHERE id=?', (parent,)).fetchone()):
                raise ResearchError('Parent must identify a different already registered trial')
            in_family = self.db.execute('SELECT DISTINCT r.trial FROM registrations r JOIN hypotheses h ON h.id=r.hypothesis WHERE h.family=?',
                                        (family.family_id,)).fetchall()
            if trial not in {r['trial'] for r in in_family} and len(in_family) >= family.max_unique_trials:
                raise ResearchError('Family unique-trial budget exhausted')
            body = canonical(spec.computation())
            prior = self.db.execute('SELECT body FROM trials WHERE id=?', (trial,)).fetchone()
            if prior and prior['body'] != body:
                raise ResearchError('Trial computation identity collision')
            self.db.execute('INSERT OR IGNORE INTO trials VALUES(?,?,?,?,?)',
                            (trial, body, spec.comparison_id(), family.max_attempts_per_trial, stamp()))
            self.db.execute('INSERT INTO registrations VALUES(?,?,?,?)', (hypothesis.hypothesis_id, trial, parent, stamp()))
            self.event('registration', {'trial': trial, 'hypothesis': hypothesis.hypothesis_id, 'parent': parent,
                                       'previously_registered_computation': prior is not None})
        return trial

    def begin(self, spec, directory, retry_of=None, *, access_bounds=('2014-01-01', '2022-12-31')):
        trial, hypothesis = spec.semantic_id(), spec.hypothesis_id
        with self.transaction():
            registered = self.db.execute('SELECT 1 FROM registrations r JOIN hypotheses h ON h.id=r.hypothesis '
                                         'WHERE r.hypothesis=? AND r.trial=? AND h.family=?',
                                         (hypothesis, trial, spec.family_id)).fetchone()
            if not registered:
                raise ResearchError('Register this computation and hypothesis before executing')
            import json
            row = self.db.execute('SELECT f.body FROM families f JOIN hypotheses h ON h.family=f.id WHERE h.id=?', (hypothesis,)).fetchone()
            limit = min(json.loads(row['body'])['max_attempts_per_trial'],
                        self.db.execute('SELECT max_attempts FROM trials WHERE id=?', (trial,)).fetchone()[0])
            attempts = self.db.execute('SELECT * FROM attempts WHERE trial=? ORDER BY created,id', (trial,)).fetchall()
            if any(r['status'] in ('running', 'completed') for r in attempts):
                raise ResearchError('Trial is active, unresolved or completed; inspect/reconcile its saved evidence')
            if len(attempts) >= limit:
                raise ResearchError('Trial attempt budget exhausted')
            if (attempts and retry_of != attempts[-1]['id']) or (not attempts and retry_of is not None):
                raise ResearchError('Retry must explicitly identify the last failed attempt of this computation')
            attempt = uuid.uuid4().hex
            self.db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)',
                            (attempt, trial, hypothesis, str(Path(directory).resolve()), 'running', retry_of,
                             stamp(), None, None, None))
            self.db.execute('INSERT INTO exposures VALUES(?,?,?,?,?,?,?,?)',
                            (attempt, trial, hypothesis, access_bounds[0], access_bounds[1], None, stamp(),
                             'development_access_potential_results_unresolved'))
            self.event('attempt_started', {'attempt': attempt, 'trial': trial, 'retry_of': retry_of})
        return attempt

    def finish(self, attempt, *, complete=None, failure=None):
        if (complete is None) == (failure is None):
            raise ResearchError('Require exactly one explicit terminal outcome')
        with self.transaction():
            row = self.db.execute('SELECT * FROM attempts WHERE id=?', (attempt,)).fetchone()
            if not row:
                raise ResearchError('Unknown registered attempt')
            target = 'completed' if complete is not None else 'failed'
            result_hash = complete['result_hash'] if complete else None
            if row['status'] != 'running':
                if row['status'] == target and row['result_hash'] == result_hash and row['failure'] == failure:
                    return
                raise ResearchError('Cannot change a recorded terminal outcome')
            if complete and (complete['trial_id'] != row['trial'] or complete['attempt_id'] != attempt
                             or complete['hypothesis_id'] != row['hypothesis'] or complete['directory'] != row['path']):
                raise ResearchError('Completion does not identify the registered attempt/path')
            self.db.execute('UPDATE attempts SET status=?,finished=?,result_hash=?,failure=? WHERE id=?',
                            (target, stamp(), result_hash, failure, attempt))
            kind=complete.get('evidence_kind','development_signal_diagnostics_published') if complete else None
            if kind not in (None,'development_signal_diagnostics_published','development_raw_share_ledger_published',
                            'development_action_accounting_continuation_published'):
                raise ResearchError('Unsupported completed evidence kind')
            self.db.execute('UPDATE exposures SET result_hash=?,kind=? WHERE attempt=?',
                            (result_hash, kind if complete else
                             'development_failed_attempt_partial_results_possible', attempt))
            self.event('attempt_'+target, {'attempt': attempt, 'result_hash': result_hash, 'failure': failure})

    def snapshot(self):
        return {table: [dict(row) for row in self.db.execute('SELECT * FROM '+table)]
                for table in ('families', 'hypotheses', 'trials', 'registrations', 'attempts', 'exposures', 'prior_exposures', 'events')}

    def note_prior_exposure(self, exposure, evidence_path):
        from .contracts import file_hash
        if file_hash(evidence_path) != exposure.evidence_sha256:
            raise ResearchError('Historical exposure evidence hash mismatch')
        with self.transaction():
            prior = self.db.execute('SELECT body FROM prior_exposures WHERE id=?', (exposure.exposure_id,)).fetchone()
            if prior:
                if prior['body'] != canonical(exposure.model_dump()):
                    raise ResearchError('Historical exposure declaration is immutable')
                return
            self.db.execute('INSERT INTO prior_exposures VALUES(?,?,?)',
                            (exposure.exposure_id, canonical(exposure.model_dump()), stamp()))
            self.event('prior_exposure_recorded', {'exposure_id': exposure.exposure_id, 'evidence_sha256': exposure.evidence_sha256})

    def verify_attempt(self, evidence):
        row = self.db.execute('SELECT * FROM attempts WHERE id=?', (evidence['attempt_id'],)).fetchone()
        if (not row or row['path'] != evidence['directory'] or row['trial'] != evidence['trial_id']
                or row['hypothesis'] != evidence['hypothesis_id']):
            raise ResearchError('Saved attempt identity/path differs from registry')
        if row['status'] == 'completed' and row['result_hash'] != evidence['result_hash']:
            raise ResearchError('Published result differs from recorded exposure')
        return dict(row)
