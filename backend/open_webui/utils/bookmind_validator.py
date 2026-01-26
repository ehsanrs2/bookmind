import json
from pathlib import Path
from typing import List, Tuple

from jsonschema import Draft202012Validator, FormatChecker


def _resolve_path(path: str | Path) -> Path:
    schema_path = Path(path)
    if schema_path.is_absolute():
        return schema_path
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / schema_path


def load_json(path: str | Path) -> dict:
    json_path = _resolve_path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate(payload: dict, schema_path: str | Path) -> Tuple[bool, List[str]]:
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    messages: List[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.path)
        if path:
            messages.append(f"{path}: {error.message}")
        else:
            messages.append(error.message)
    return (len(messages) == 0, messages)


def validate_scope(
    scope: dict,
    schema_path: str | Path = "backend/open_webui/schemas/bookmind/scope.schema.json",
) -> Tuple[bool, List[str]]:
    return _validate(scope, schema_path)


def validate_chunk_metadata(
    meta: dict,
    schema_path: str | Path = "backend/open_webui/schemas/bookmind/chunk-metadata.schema.json",
) -> Tuple[bool, List[str]]:
    return _validate(meta, schema_path)
