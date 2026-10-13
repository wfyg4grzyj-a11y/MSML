"""Versioned, typed Parquet storage for the direct Sharadar full bundle."""

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb

from .sharadar import SharadarError, file_hash, table_name


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def identifier(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SharadarError("Invalid column name")
    return '"' + name + '"'


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SharadarStore:
    def __init__(self, root: Path, *, memory_limit="4GB", threads=4):
        if not re.fullmatch(r"[1-9][0-9]*(MB|GB)", memory_limit) or not 1 <= threads <= 32:
            raise SharadarError("Invalid memory limit or thread count")
        self.root = Path(root).resolve()
        self.memory_limit, self.threads = memory_limit, threads

    @contextmanager
    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / "duckdb_temp"
        temporary.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as spill:
            with duckdb.connect(config={"memory_limit": self.memory_limit, "threads": str(self.threads),
                                        "temp_directory": spill}) as db:
                yield db

    @contextmanager
    def _writer_lock(self):
        """OS locks release on process exit, so interrupted runs leave no stale lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "sync.lock").open("a+b") as lock:
            lock.seek(0, os.SEEK_END)
            if not lock.tell():
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                release = lambda: fcntl.flock(lock, fcntl.LOCK_UN)
            try:
                acquire()
            except OSError:
                raise SharadarError("Another sync is already writing this Sharadar storage root") from None
            try:
                yield
            finally:
                lock.seek(0)
                release()

    def latest(self, table):
        table = table_name(table)
        pointer = self.root / "tables" / table / "latest.json"
        if not pointer.exists():
            return None
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            version = payload["version"]
            if not re.fullmatch(r"[a-f0-9]{32}", version):
                raise ValueError
            base = pointer.parent / version
            manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
            if manifest["table"] != table or not (base / "data.parquet").is_file():
                raise ValueError
            return manifest | {"version": version, "path": str(base / "data.parquet")}
        except (OSError, ValueError, KeyError, TypeError):
            raise SharadarError("Invalid local Sharadar snapshot") from None

    def summary(self, tables):
        return [{"table": table_name(t), "snapshot": self.latest(t)} for t in tables]

    def sync(self, client, table, *, force=False, progress=None, event=None):
        with self._writer_lock():
            return self._sync(client, table, force=force, progress=progress, event=event)

    def _sync(self, client, table, *, force=False, progress=None, event=None):
        table = table_name(table)
        metadata = client.status(table)
        current = self.latest(table)
        if not force and current and current["source"] == metadata:
            return {"table": table, "status": "unchanged", "rows": current["rows"]}
        schema = client.schema(table)
        # A source fingerprint allows completed archives and partial downloads to be reused.
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
        archive = self.root / "raw" / table / (fingerprint + ".zip")
        if not archive.exists() or archive.stat().st_size != metadata["size"]:
            download = client.download(table, metadata, archive, progress=progress)
        else:
            download = {"sha256": file_hash(archive), "bytes": archive.stat().st_size}
        if event:
            event(table, "validating")
        result = self.publish(table, archive, schema, metadata, download)
        return {"table": table, "status": "updated", "rows": result["rows"],
                "start_date": result.get("start_date"), "end_date": result.get("end_date")}

    def publish(self, table, archive, schema, metadata, download):
        """Publish only after ZIP CRC, schema, type and key validation succeed."""
        table = table_name(table)
        version = uuid.uuid4().hex
        stage = self.root / "staging" / version
        if stage.resolve() != stage:
            raise SharadarError("Staging path must stay within the configured storage root")
        stage.mkdir(parents=True)
        csv_path = stage / "source.csv"
        parquet = stage / "data.parquet"
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = [entry for entry in bundle.infolist() if not entry.is_dir()]
                if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
                    raise SharadarError("Expected a single CSV inside the Sharadar ZIP")
                # Copy to our own filename; no archive paths are ever extracted.
                with bundle.open(members[0]) as source, csv_path.open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            columns = schema["columns"]
            names = [column["name"] for column in columns]
            with csv_path.open(encoding="utf-8-sig", newline="") as source:
                header = next(csv.reader(source), [])
            if header != names:
                raise SharadarError("Bulk CSV columns differ from the provider schema")
            fields = ", ".join(f"CAST({identifier(c['name'])} AS {c['type']}) AS {identifier(c['name'])}" for c in columns)
            # Schema types are restricted by parse_schema, but validate callers too.
            if any(c["type"] not in {"VARCHAR", "DATE", "BIGINT", "INTEGER", "DOUBLE", "BOOLEAN"} for c in columns):
                raise SharadarError("Invalid schema type")
            with self._connect() as db:
                # DuckDB's DDL does not support bound parameters; quote only our local path.
                csv_literal = "'" + str(csv_path).replace("'", "''") + "'"
                db.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv({csv_literal}, header=true, all_varchar=true, nullstr='', strict_mode=true)")
                db.execute(f"COPY (SELECT {fields} FROM raw) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(parquet)])
                db.from_parquet(str(parquet)).create_view("snapshot")
                rows = db.execute("SELECT count(*) FROM snapshot").fetchone()[0]
                if not rows:
                    raise SharadarError("Empty Sharadar snapshot")
                primary = schema["primary_key"]
                if not primary or not set(primary).issubset(names):
                    raise SharadarError("Missing Sharadar primary key")
                keys = ", ".join(identifier(name) for name in primary)
                duplicate = db.execute(f"SELECT 1 FROM snapshot GROUP BY {keys} HAVING count(*) > 1 LIMIT 1").fetchone()
                if duplicate:
                    raise SharadarError("Duplicate primary keys in Sharadar snapshot")
                # Some action key strings are intentionally empty; retain provider nulls.
                required = [identifier(c["name"]) + " IS NULL" for c in columns
                            if c["required"] and (c["type"] != "VARCHAR" or c["name"] not in {"name", "contraticker", "contraname"})]
                if required and db.execute("SELECT 1 FROM snapshot WHERE " + " OR ".join(required) + " LIMIT 1").fetchone():
                    raise SharadarError("Missing required key in snapshot")
                nonfinite = [f"NOT isfinite({identifier(c['name'])})" for c in columns if c["type"] == "DOUBLE"]
                if nonfinite and db.execute("SELECT 1 FROM snapshot WHERE " + " OR ".join(nonfinite) + " LIMIT 1").fetchone():
                    raise SharadarError("Nonfinite numeric value in snapshot")
                bounds = {}
                if "date" in names:
                    low, high = db.execute('SELECT min(TRY_CAST("date" AS DATE)), max(TRY_CAST("date" AS DATE)) FROM snapshot').fetchone()
                    bounds = {"start_date": str(low) if low else None, "end_date": str(high) if high else None}
            manifest = {"schema_version": 1, "provider": "sharadar.com", "table": table,
                        "source": metadata, "retrieved_at": utc_now(), "rows": rows, "schema": schema,
                        "archive_sha256": download["sha256"], "archive_bytes": download["bytes"],
                        "parquet_sha256": file_hash(parquet), **bounds}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            csv_path.unlink()
            destination = self.root / "tables" / table / version
            if stage.resolve() != stage or destination.resolve() != destination:
                raise SharadarError("Snapshot paths must stay within the configured storage root")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, destination)
            atomic_json(destination.parent / "latest.json", {"version": version})
            return manifest
        except zipfile.BadZipFile:
            # Retrying must fetch a fresh archive instead of reusing a corrupt ZIP.
            Path(archive).unlink(missing_ok=True)
            raise SharadarError("ZIP integrity validation failed; retry to download a fresh archive") from None
        except (duckdb.Error, UnicodeError, csv.Error):
            raise SharadarError("ZIP, CSV or typed-data validation failed; previous snapshot retained") from None
        finally:
            if stage.exists():
                resolved_stage = stage.resolve()
                if resolved_stage.parent != self.root / "staging" or resolved_stage.name != version:
                    raise SharadarError("Refusing to remove an unexpected staging directory")
                shutil.rmtree(resolved_stage)

    def export(self, table, destination, *, tickers=None, start=None, end=None, dimension=None, columns=None):
        """Offline filtered CSV with provenance; never overwrite an existing export."""
        table = table_name(table)
        snapshot = self.latest(table)
        if snapshot is None:
            raise SharadarError("Table has not been downloaded yet")
        names = [c["name"] for c in snapshot["schema"]["columns"]]
        selected = columns or names
        if not selected or not set(selected).issubset(names) or len(set(selected)) != len(selected):
            raise SharadarError("Unknown or duplicate export columns")
        conditions, parameters = [], [snapshot["path"]]
        if tickers:
            if "ticker" not in names:
                raise SharadarError("This table has no ticker column")
            conditions.append('"ticker" IN (' + ",".join("?" for _ in tickers) + ")")
            parameters.extend(tickers)
        if start or end:
            if "date" not in names:
                raise SharadarError("This table has no date column")
            for value, operator in ((start, ">="), (end, "<=")):
                if value:
                    date.fromisoformat(str(value))
                    conditions.append(f'TRY_CAST("date" AS DATE) {operator} CAST(? AS DATE)')
                    parameters.append(str(value))
            if start and end and str(start) > str(end):
                raise SharadarError("Start date must precede end date")
        if dimension:
            if "dimension" not in names:
                raise SharadarError("This table has no fundamentals dimension")
            conditions.append('"dimension" = ?')
            parameters.append(dimension)
        query = "SELECT " + ", ".join(identifier(name) for name in selected) + " FROM read_parquet(?)"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY " + ", ".join(identifier(name) for name in snapshot["schema"]["primary_key"])
        destination = Path(destination).resolve()
        sidecar = destination.with_suffix(destination.suffix + ".manifest.json")
        if destination.exists() or sidecar.exists():
            raise SharadarError("Export already exists; choose a new filename")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / ("." + uuid.uuid4().hex + ".csv")
        created = False
        manifest_created = False
        try:
            with self._connect() as db:
                db.sql(query, params=parameters).write_csv(str(temporary), header=True)
            with destination.open("xb") as target, temporary.open("rb") as source:
                created = True
                shutil.copyfileobj(source, target)
            manifest = {"provider": "sharadar.com", "table": table, "source_version": snapshot["version"],
                        "source": snapshot["source"], "created_at": utc_now(), "sha256": file_hash(destination),
                        "filters": {"tickers": tickers, "start": start, "end": end, "dimension": dimension}, "columns": selected}
            with sidecar.open("x", encoding="utf-8") as output:
                manifest_created = True
                json.dump(manifest, output, indent=2)
            return manifest
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            if manifest_created:
                sidecar.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
