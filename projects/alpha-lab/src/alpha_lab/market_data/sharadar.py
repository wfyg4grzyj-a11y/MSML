"""Direct sharadar.com API and resumable full-history ZIP downloads.

The provider requires query-string authentication. Never log request URLs,
response bodies, signed download links, or underlying HTTP exceptions.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

TABLES = (
    "descriptions", "tickers", "sp500", "actions", "events", "metrics",
    "fundamentals", "daily", "stocks", "funds", "insiders", "holdings",
    "holdings_ticker", "holdings_investor",
)
ALIASES = {"SF1": "fundamentals", "SEP": "stocks", "SFP": "funds", "SF2": "insiders",
           "SF3": "holdings", "SF3A": "holdings_ticker", "SF3B": "holdings_investor"}


class SharadarError(RuntimeError):
    """Sanitized, safe-to-display error."""


def table_name(value: str) -> str:
    result = ALIASES.get(value.upper(), value.lower())
    if result not in TABLES:
        raise SharadarError("Unsupported Sharadar table")
    return result


def parse_schema(text: str, table: str) -> dict:
    """Parse only field/type declarations; never execute upstream DDL."""
    table = table_name(table)
    match = re.search(r"CREATE TABLE IF NOT EXISTS " + re.escape(table) + r"\s*\((.*?)\n\);", text, re.S | re.I)
    if not match:
        raise SharadarError("Unsupported Sharadar schema format")
    columns, primary = [], []
    types = {"text": "VARCHAR", "bigint": "BIGINT", "integer": "INTEGER", "date": "DATE",
             "double precision": "DOUBLE", "boolean": "BOOLEAN"}
    for line in match.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        if line.upper().startswith("PRIMARY KEY"):
            key = re.fullmatch(r'PRIMARY KEY \(([a-z0-9_, " ]+)\)', line, re.I)
            if not key:
                raise SharadarError("Unsupported Sharadar primary key")
            primary = [name.strip().strip('"') for name in key.group(1).split(",")]
            continue
        field = re.fullmatch(r'("?[a-z][a-z0-9_]*"?) (text|bigint|integer|date|double precision|boolean)( NOT NULL)?', line, re.I)
        if not field:
            raise SharadarError("Unsupported Sharadar column type")
        name, kind, required = field.groups()
        columns.append({"name": name.lower().strip('"'), "type": types[kind.lower()], "required": bool(required)})
    names = [column["name"] for column in columns]
    if not primary:
        key = re.search(r'CREATE UNIQUE INDEX IF NOT EXISTS [a-z0-9_]+ ON public\.' + re.escape(table) + r' USING btree \(([a-z0-9_, " ]+)\);', text, re.I)
        if key:
            primary = [name.strip().strip('"') for name in key.group(1).split(",")]
    if not columns or len(set(names)) != len(names) or not primary or not set(primary).issubset(names):
        raise SharadarError("Invalid Sharadar schema")
    return {"columns": columns, "primary_key": primary}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SharadarClient:
    def __init__(self, api_key: str, *, requests_per_minute=30, retries=5,
                 transport=None, sleep=time.sleep, clock=time.monotonic):
        if not api_key or not api_key.strip():
            raise SharadarError("Set SHARADAR_API_KEY in your environment or local .env")
        if not 0 < requests_per_minute <= 600 or not 0 <= retries <= 8:
            raise SharadarError("Invalid Sharadar rate or retry count")
        self._key = api_key.strip()
        self._http = httpx.Client(timeout=httpx.Timeout(60, connect=30), follow_redirects=False, transport=transport)
        self._sleep, self._clock = sleep, clock
        self._interval, self._last = 60 / requests_per_minute, None
        self.retries = retries

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._http.close()

    def _request(self, path, params=None, *, stream=False):
        for attempt in range(self.retries + 1):
            if self._last is not None:
                self._sleep(max(0, self._interval - (self._clock() - self._last)))
            self._last = self._clock()
            query = dict(params or {})
            if path.startswith("data/"):
                query["api_key"] = self._key
            request = self._http.build_request("GET", "https://api.sharadar.com/v1.0/" + path, params=query)
            try:
                response = self._http.send(request, stream=stream)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise SharadarError("Sharadar connection failed after retries") from None
                self._sleep(min(30, 2**attempt))
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                delay = min(30, 2**attempt)
                try:
                    delay = max(0, min(60, float(response.headers.get("Retry-After", delay))))
                except ValueError:
                    pass
                status = response.status_code
                response.close()
                if attempt == self.retries:
                    raise SharadarError(f"Sharadar HTTP {status}; retry later")
                self._sleep(delay)
                continue
            if response.status_code not in (200, 301, 302, 303, 307, 308):
                status = response.status_code
                response.close()
                raise SharadarError(f"Sharadar HTTP {status}; check the direct API key and table entitlement")
            return response

    def schema(self, table: str) -> dict:
        response = self._request("schema/" + table_name(table), {"format": "postgres"})
        try:
            if response.status_code != 200:
                raise SharadarError("Unexpected schema redirect")
            return parse_schema(response.text, table)
        finally:
            response.close()

    def status(self, table: str) -> dict:
        table = table_name(table)
        response = self._request("data/" + table, {"status": "True", "years": "full"})
        try:
            payload = response.json()
            files = payload.get("files", [])
            selected = next((entry for entry in files if entry.get("history") == "full"), payload if not files else None)
            if not isinstance(selected, dict) or selected.get("available") is False:
                raise SharadarError("Full-history bulk file is unavailable")
            size = selected["size"]
            modified = selected["modified"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not isinstance(modified, str):
                raise SharadarError("Invalid Sharadar bulk metadata")
            return {"table": table, "size": size, "modified": modified, "history": "full"}
        except (ValueError, KeyError, TypeError, AttributeError):
            raise SharadarError("Invalid Sharadar bulk metadata") from None
        finally:
            response.close()

    def _download_link(self, table: str) -> str:
        response = self._request("data/" + table, {"years": "full"}, stream=True)
        try:
            if response.status_code not in (301, 302, 303, 307, 308):
                raise SharadarError("Sharadar did not issue a bulk download link; check entitlement")
            if not response.headers.get("Location"):
                raise SharadarError("Missing bulk download link")
            link = urljoin("https://api.sharadar.com", response.headers["Location"])
            parsed = urlsplit(link)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise SharadarError("Invalid bulk download link")
            return link
        finally:
            response.close()

    def download(self, table: str, metadata: dict, destination: Path, *, progress=None) -> dict:
        """Resume only the same source version; reissue expiring signed links."""
        table = table_name(table)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        state_path = partial.with_suffix(partial.suffix + ".json")
        source = {"table": table, "size": metadata["size"], "modified": metadata["modified"]}
        state = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                pass
        if state.get("source") != source:
            partial.unlink(missing_ok=True)
            state = {"source": source}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        for attempt in range(self.retries + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == metadata["size"]:
                partial.replace(destination)
                state_path.unlink(missing_ok=True)
                return {"sha256": file_hash(destination), "bytes": offset}
            if offset > metadata["size"]:
                partial.unlink()
                offset = 0
            link = self._download_link(table)
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if state.get("etag"):
                    headers["If-Range"] = state["etag"]
            try:
                # No API key is attached to the storage request or forwarded in headers.
                with self._http.stream("GET", link, headers=headers) as response:
                    if response.status_code not in (200, 206):
                        raise SharadarError("Bulk storage request failed; retry later")
                    if response.status_code == 206:
                        content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not content_range or int(content_range[1]) != offset or int(content_range[3]) != metadata["size"]:
                            raise SharadarError("Invalid bulk byte range")
                        if state.get("etag") and response.headers.get("ETag") != state["etag"]:
                            partial.unlink(missing_ok=True)
                            state.pop("etag", None)
                            raise SharadarError("Bulk source changed; restarting download")
                    elif offset:
                        offset = 0  # Server ignored Range or If-Range detected a new object.
                    state["etag"] = response.headers.get("ETag")
                    state_path.write_text(json.dumps(state), encoding="utf-8")
                    last_notice = 0
                    with partial.open("ab" if offset else "wb") as output:
                        for block in response.iter_raw(1024 * 1024):
                            output.write(block)
                            offset += len(block)
                            if offset > metadata["size"]:
                                raise SharadarError("Bulk download exceeds expected size")
                            if progress and self._clock() - last_notice >= 5:
                                progress(table, offset, metadata["size"])
                                last_notice = self._clock()
                    if offset != metadata["size"]:
                        raise SharadarError("Bulk download is incomplete")
            except (httpx.TransportError, SharadarError):
                if attempt == self.retries:
                    raise SharadarError("Bulk download failed; partial data retained for retry") from None
                self._sleep(min(30, 2**attempt))
                continue
            partial.replace(destination)
            state_path.unlink(missing_ok=True)
            return {"sha256": file_hash(destination), "bytes": offset}
