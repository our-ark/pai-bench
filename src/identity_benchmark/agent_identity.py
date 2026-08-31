from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
import json
import re
from typing import Mapping, TypeAlias

from identity_benchmark.json_types import JsonValue


AgentIdentity: TypeAlias = dict[str, JsonValue]
SCHEMA_RESOURCE = ("schemas", "ai-agent-identity.schema.json")
SCHEMA_ID = "https://our-ark.github.io/schemas/ai-agent-identity.schema.json"
_DATE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_SUPPORTED_KEYWORDS = {
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "title",
    "description",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "const",
    "minLength",
    "minProperties",
    "minItems",
    "uniqueItems",
    "items",
    "format",
}


class AgentIdentityError(ValueError):
    """Raised when a portable Agent Identity violates its JSON Schema."""


def parse_agent_identity(
    value: object,
    *,
    label: str = "Agent Identity",
) -> AgentIdentity:
    """Validate and copy one portable Agent Identity document."""
    schema = agent_identity_schema()
    _validate(value, schema, schema, label)
    if not isinstance(value, Mapping):  # Narrowed by schema validation.
        raise AgentIdentityError(f"{label} must be an object")
    return deepcopy(dict(value))  # type: ignore[return-value]


@lru_cache(maxsize=1)
def agent_identity_schema() -> dict[str, JsonValue]:
    """Load the packaged runtime copy of the public identity schema."""
    resource = files("identity_benchmark").joinpath(*SCHEMA_RESOURCE)
    try:
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AgentIdentityError(
            f"could not load packaged Agent Identity schema: {error}"
        ) from error
    if not isinstance(value, dict) or value.get("$id") != SCHEMA_ID:
        raise AgentIdentityError("packaged Agent Identity schema has an invalid $id")
    return value


def _validate(
    value: object,
    schema: Mapping[str, object],
    root: Mapping[str, object],
    path: str,
) -> None:
    unsupported = sorted(set(schema) - _SUPPORTED_KEYWORDS)
    if unsupported:
        raise AgentIdentityError(
            f"{path}: runtime validator does not support schema keyword(s): "
            + ", ".join(unsupported)
        )
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise AgentIdentityError(f"{path}: schema $ref must be text")
        _validate(value, _resolve_reference(reference, root, path), root, path)
        return
    if "const" in schema and value != schema["const"]:
        raise AgentIdentityError(f"{path} must equal {schema['const']!r}")
    expected_type = schema.get("type")
    if expected_type == "object":
        _validate_object(value, schema, root, path)
    elif expected_type == "array":
        _validate_array(value, schema, root, path)
    elif expected_type == "string":
        _validate_string(value, schema, path)
    elif expected_type is not None:
        raise AgentIdentityError(
            f"{path}: runtime validator does not support type {expected_type!r}"
        )


def _validate_object(
    value: object,
    schema: Mapping[str, object],
    root: Mapping[str, object],
    path: str,
) -> None:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AgentIdentityError(f"{path} must be an object")
    minimum = schema.get("minProperties", 0)
    if not isinstance(minimum, int):
        raise AgentIdentityError(f"{path}: schema minProperties must be an integer")
    if len(value) < minimum:
        raise AgentIdentityError(f"{path} must contain at least {minimum} properties")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(item, str) for item in required
    ):
        raise AgentIdentityError(f"{path}: schema required must be a string array")
    missing = sorted(set(required) - set(value))
    if missing:
        raise AgentIdentityError(
            f"{path} is missing required properties: {', '.join(missing)}"
        )
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise AgentIdentityError(f"{path}: schema properties must be an object")
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
        item_path = f"{path}.{key}"
        child = properties.get(key)
        if child is not None:
            if not isinstance(child, Mapping):
                raise AgentIdentityError(f"{item_path}: property schema is invalid")
            _validate(item, child, root, item_path)
        elif additional is False:
            raise AgentIdentityError(f"{path} contains unsupported property {key!r}")
        elif isinstance(additional, Mapping):
            _validate(item, additional, root, item_path)
        elif additional is not True:
            raise AgentIdentityError(
                f"{path}: schema additionalProperties must be boolean or object"
            )


def _validate_array(
    value: object,
    schema: Mapping[str, object],
    root: Mapping[str, object],
    path: str,
) -> None:
    if not isinstance(value, list):
        raise AgentIdentityError(f"{path} must be an array")
    minimum = schema.get("minItems", 0)
    if not isinstance(minimum, int):
        raise AgentIdentityError(f"{path}: schema minItems must be an integer")
    if len(value) < minimum:
        raise AgentIdentityError(f"{path} must contain at least {minimum} items")
    if schema.get("uniqueItems", False):
        encoded = [
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in value
        ]
        if len(encoded) != len(set(encoded)):
            raise AgentIdentityError(f"{path} must contain unique items")
    item_schema = schema.get("items")
    if item_schema is not None:
        if not isinstance(item_schema, Mapping):
            raise AgentIdentityError(f"{path}: schema items must be an object")
        for index, item in enumerate(value):
            _validate(item, item_schema, root, f"{path}[{index}]")


def _validate_string(
    value: object,
    schema: Mapping[str, object],
    path: str,
) -> None:
    if not isinstance(value, str):
        raise AgentIdentityError(f"{path} must be text")
    minimum = schema.get("minLength", 0)
    if not isinstance(minimum, int):
        raise AgentIdentityError(f"{path}: schema minLength must be an integer")
    if len(value) < minimum:
        raise AgentIdentityError(f"{path} must contain at least {minimum} characters")
    format_name = schema.get("format")
    if format_name is not None:
        if format_name != "date-time":
            raise AgentIdentityError(
                f"{path}: runtime validator does not support format {format_name!r}"
            )
        if not _valid_date_time(value):
            raise AgentIdentityError(f"{path} must be an RFC 3339 date-time")


def _resolve_reference(
    reference: str,
    root: Mapping[str, object],
    path: str,
) -> Mapping[str, object]:
    if not reference.startswith("#/"):
        raise AgentIdentityError(
            f"{path}: runtime validator supports only local schema references"
        )
    current: object = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise AgentIdentityError(f"{path}: unresolved schema reference {reference}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise AgentIdentityError(f"{path}: schema reference {reference} is not an object")
    return current


def _valid_date_time(value: str) -> bool:
    if not _DATE_TIME.fullmatch(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None
