"""Shared PROV-O helpers and namespace configuration.

Used by all PROV document builders (user, dataset, training run).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import Any


# ──────────────────────────────────────────────
# OpenProvenance / CPM namespace URIs
# ──────────────────────────────────────────────

_DEFAULT_PROV_PREFIXES: dict[str, str] = {
    "storage": "http://localhost:8083/api/v1/documents/",
    "meta": "http://localhost:8083/api/v1/documents/meta/",
    "schema": "https://schema.org/",
    "cpm": "https://www.commonprovenancemodel.org/cpm-namespace-v1-0/",
    "blank": "https://openprovenance.org/blank/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "gen": "gen/",
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "sosa": "http://www.w3.org/ns/sosa/",
}


def get_prov_prefixes(override: dict[str, str] | None = None) -> dict[str, str]:
    """Return PROV prefix map.

    Priority: explicit override > ``PROV_BASE_URI`` env var > defaults.
    """
    if override:
        return {**_DEFAULT_PROV_PREFIXES, **override}
    env_json = os.environ.get("PROV_BASE_URI", "")
    if env_json:
        try:
            parsed = json.loads(env_json)
            if not isinstance(parsed, dict):
                raise TypeError("expected a JSON object")
            merged = {**_DEFAULT_PROV_PREFIXES, **parsed}
            return merged
        except (json.JSONDecodeError, TypeError):
            pass
    return _DEFAULT_PROV_PREFIXES


# ──────────────────────────────────────────────
# Small helpers used inside PROV documents
# ──────────────────────────────────────────────


def _safe_id(name: str) -> str:
    """Sanitise a name so it can be used as a PROV identifier fragment."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _qualified(prefix: str, local: str) -> str:
    return f"{prefix}:{local}"


def _typed_value(value: Any) -> list[str]:
    return [str(value)]


def _qualified_name(type_prefix: str, type_local: str) -> dict[str, str]:
    return {"type": "prov:QUALIFIED_NAME", "$": f"{type_prefix}:{type_local}"}


def _iso_timestamp(ts_ms: int | None = None) -> str:
    if ts_ms is not None:
        dt = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.UTC)
    else:
        dt = _dt.datetime.now(_dt.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
