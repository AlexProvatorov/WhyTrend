"""Build a Pipeline from CLI arguments and installed plugins."""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from whytrend.core.protocols import Collector, Detector, Explainer, Ranker, Source
from whytrend.pipeline import Pipeline
from whytrend.plugins import PluginError, list_plugins, load_plugin

COLLECTOR_ALIASES: dict[str, str] = {"hn": "hacker_news"}

_COMPONENT_SETTING = re.compile(
    r"^(?P<target>detector|explainer|ranker|source|provider)"
    r"\.(?P<param>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)
_COLLECTOR_SETTING = re.compile(
    r"^collector\.(?P<collector>[A-Za-z_][A-Za-z0-9_]*)"
    r"\.(?P<param>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)
_SCALAR_TYPES = frozenset({str, int, float, bool})
_SETTING_USAGE = (
    "expected detector.<param>=<value>, explainer.<param>=<value>, "
    "ranker.<param>=<value>, source.<param>=<value>, provider.<param>=<value>, "
    "or collector.<name>.<param>=<value>"
)
_Kind = Literal["bool", "int", "float", "str", "str_list", "dict"]


class UsageError(Exception):
    """Invalid CLI arguments that still need a clear message and exit code 2."""

    code = 2


@dataclass(frozen=True)
class Setting:
    """One ``--set`` override."""

    target: str
    param: str
    value: str
    collector: str | None = None


@dataclass(frozen=True)
class _Field:
    kind: _Kind
    optional: bool


def resolve_collector(name: str) -> str:
    """Map a collector alias to its entry-point name."""
    return COLLECTOR_ALIASES.get(name, name)


def required_param_names(cls: type[Any]) -> list[str]:
    """Return ``__init__`` parameters that have no default."""
    return [
        name
        for name, param, _annotation in _init_parameters(cls)
        if param.default is inspect.Parameter.empty
    ]


def format_plugin_list(group: str) -> str:
    """Return help text for every plugin, noting required constructor arguments."""
    plugins = list_plugins(group)
    labels = [_plugin_label(group, name, cls) for name, cls in sorted(plugins.items())]
    return ", ".join(labels) if labels else "(none)"


def require_plugin(group: str, name: str) -> str:
    """Return ``name`` when the plugin exists."""
    try:
        load_plugin(group, name)
    except PluginError as exc:
        raise UsageError(str(exc)) from exc
    return name


def parse_collector_list(value: str) -> list[str]:
    """Parse a comma-separated collector list, resolving aliases."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        msg = "collectors must be a comma-separated list of plugin names"
        raise UsageError(msg)
    resolved: list[str] = []
    seen: set[str] = set()
    for part in parts:
        name = require_plugin("collectors", resolve_collector(part))
        if name not in seen:
            seen.add(name)
            resolved.append(name)
    return resolved


def parse_setting(value: str) -> Setting:
    """Parse one ``--set`` value."""
    text = value.strip()
    component = _COMPONENT_SETTING.fullmatch(text)
    if component is not None:
        return Setting(
            target=component.group("target"),
            param=component.group("param"),
            value=component.group("value"),
        )
    collector = _COLLECTOR_SETTING.fullmatch(text)
    if collector is not None:
        return Setting(
            target="collector",
            param=collector.group("param"),
            value=collector.group("value"),
            collector=collector.group("collector"),
        )
    raise UsageError(_SETTING_USAGE)


def build_pipeline(args: Any) -> Pipeline:
    """Assemble a pipeline from parsed ``analyze`` arguments."""
    settings = list(args.settings or [])
    collectors = list(args.collectors or [])
    ranker_name = cast(str | None, args.ranker)
    provider_name = cast(str | None, args.provider)
    window_days = cast(int, args.window_days)
    if window_days < 1:
        msg = "--window-days must be >= 1"
        raise UsageError(msg)
    _validate_settings(settings, collectors, ranker_name, provider_name)

    pipeline = (
        Pipeline(window_days=window_days)
        .add_source(_build_source(args, settings))
        .add_detector(
            cast(
                Detector,
                _instantiate("detectors", cast(str, args.detector), _params(settings, "detector")),
            )
        )
        .add_explainer(_build_explainer(args, settings))
    )
    for name in collectors:
        pipeline.add_collector(
            cast(
                Collector,
                _instantiate("collectors", name, _params(settings, "collector", name)),
            )
        )
    if ranker_name is not None:
        pipeline.add_ranker(
            cast(
                Ranker,
                _instantiate("rankers", ranker_name, _params(settings, "ranker")),
            )
        )
    return pipeline


def _validate_settings(
    settings: list[Setting],
    collectors: list[str],
    ranker_name: str | None,
    provider_name: str | None,
) -> None:
    selected = set(collectors)
    for setting in settings:
        if setting.target == "collector":
            collector = resolve_collector(setting.collector or "")
            if collector not in selected:
                msg = f"--set refers to collector {collector!r}, which is not in --collectors"
                raise UsageError(msg)
        elif setting.target == "ranker" and ranker_name is None:
            msg = f"--set ranker.{setting.param}={setting.value} requires --ranker"
            raise UsageError(msg)
        elif setting.target == "provider" and provider_name is None:
            msg = f"--set provider.{setting.param}={setting.value} requires --provider"
            raise UsageError(msg)


def _build_source(args: Any, settings: list[Setting]) -> Source:
    trends = cast(str | None, args.trends)
    csv_path = cast(Path | None, args.csv)
    parquet_path = cast(Path | None, args.parquet)
    keyword = cast(str | None, args.keyword)
    if trends is not None:
        if keyword is not None and keyword != trends:
            msg = "--keyword must match --trends"
            raise UsageError(msg)
        return _trends_source(trends, settings)
    if csv_path is not None:
        return _file_source("csv", csv_path, args, settings)
    if parquet_path is not None:
        return _file_source("parquet", parquet_path, args, settings)
    msg = "provide --csv, --parquet, or --trends"
    raise UsageError(msg)


def _trends_source(keyword: str, settings: list[Setting]) -> Source:
    cls = load_plugin("sources", "google_trends")
    return cast(
        Source,
        _construct(
            cls,
            _params(settings, "source"),
            positional=(keyword,),
            provided={"keyword"},
        ),
    )


def _file_source(plugin: str, path: Path, args: Any, settings: list[Setting]) -> Source:
    cls = load_plugin("sources", plugin)
    raw = _params(settings, "source")
    overrides = _coerce(cls, raw)
    keyword = cast(str | None, args.keyword)
    if keyword is not None and overrides.get("keyword") not in {None, keyword}:
        msg = "--set source.keyword must match --keyword"
        raise UsageError(msg)
    preset: dict[str, object] = {
        "time_col": args.time_col,
        "value_col": args.value_col,
    }
    if keyword is not None:
        preset["keyword"] = keyword
    preset.update(overrides)
    return cast(
        Source,
        _construct(cls, {}, positional=(path,), provided={"path"}, preset=preset),
    )


def _build_explainer(args: Any, settings: list[Setting]) -> Explainer:
    name = cast(str, args.explainer)
    provider_name = cast(str | None, args.provider)
    if provider_name is not None and name != "llm":
        msg = "--provider is only valid with --explainer llm"
        raise UsageError(msg)
    raw = _params(settings, "explainer")
    if name != "llm":
        return cast(Explainer, _instantiate("explainers", name, raw))
    if provider_name is None:
        available = ", ".join(sorted(list_plugins("llm"))) or "(none)"
        msg = f"--explainer llm requires --provider; available: {available}"
        raise UsageError(msg)
    provider = _instantiate("llm", provider_name, _params(settings, "provider"))
    explainer_cls = load_plugin("explainers", "llm")
    return cast(
        Explainer,
        _construct(explainer_cls, raw, positional=(provider,), provided={"provider"}),
    )


def _instantiate(group: str, name: str, raw: Mapping[str, str]) -> object:
    return _construct(load_plugin(group, name), raw)


def _construct(
    cls: type[Any],
    raw: Mapping[str, str],
    *,
    positional: tuple[object, ...] = (),
    provided: set[str] | None = None,
    preset: Mapping[str, object] | None = None,
) -> object:
    satisfied = set(provided or ())
    overlap = sorted(name for name in raw if name in satisfied)
    if overlap:
        listed = ", ".join(repr(name) for name in overlap)
        msg = f"parameter {listed} is already set by a CLI flag"
        raise UsageError(msg)
    kwargs = dict(preset or {})
    kwargs.update(_coerce(cls, raw))
    missing = [
        name for name in required_param_names(cls) if name not in kwargs and name not in satisfied
    ]
    if missing:
        joined = ", ".join(missing)
        msg = f"{cls.__name__} requires constructor arguments: {joined}"
        raise UsageError(msg)
    return cast(object, cls(*positional, **kwargs))


def _params(
    settings: list[Setting],
    target: str,
    collector: str | None = None,
) -> dict[str, str]:
    found: dict[str, str] = {}
    for setting in settings:
        if setting.target != target:
            continue
        if target == "collector" and resolve_collector(setting.collector or "") != collector:
            continue
        found[setting.param] = setting.value
    return found


def _coerce(cls: type[Any], raw: Mapping[str, str]) -> dict[str, object]:
    settable = _settable_fields(cls)
    blocked = _blocked_params(cls)
    unknown = [name for name in raw if name not in settable and name not in blocked]
    valid = ", ".join(sorted(settable)) or "(none)"
    if unknown:
        listed = ", ".join(repr(name) for name in sorted(unknown))
        msg = f"unknown parameter {listed} for {cls.__name__}; available: {valid}"
        raise UsageError(msg)
    rejected = [name for name in raw if name in blocked]
    if rejected:
        listed = ", ".join(repr(name) for name in sorted(rejected))
        msg = f"parameter {listed} cannot be set from the CLI; available: {valid}"
        raise UsageError(msg)
    return {name: _coerce_value(value, settable[name], name) for name, value in raw.items()}


def _coerce_value(raw: str, field: _Field, param: str) -> object:
    if field.optional and raw.lower() == "none":
        return None
    if field.kind == "str_list":
        return _parse_str_list(raw, param)
    if field.kind == "dict":
        return _parse_dict(raw, param)
    if field.kind == "bool":
        lowered = raw.lower()
        if lowered not in {"true", "false"}:
            msg = f"parameter {param!r} expects true or false, got {raw!r}"
            raise UsageError(msg)
        return lowered == "true"
    if field.kind == "int":
        try:
            return int(raw, 10)
        except ValueError:
            msg = f"parameter {param!r} expects an integer, got {raw!r}"
            raise UsageError(msg) from None
    if field.kind == "float":
        try:
            number = float(raw)
        except ValueError:
            msg = f"parameter {param!r} expects a number, got {raw!r}"
            raise UsageError(msg) from None
        if not math.isfinite(number):
            msg = f"parameter {param!r} expects a finite number, got {raw!r}"
            raise UsageError(msg)
        return number
    return raw


def _parse_str_list(raw: str, param: str) -> list[str]:
    if raw.startswith("["):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"parameter {param!r} expects a JSON list of strings, got {raw!r}"
            raise UsageError(msg) from exc
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            msg = f"parameter {param!r} expects a JSON list of strings, got {raw!r}"
            raise UsageError(msg)
        return cast(list[str], value)
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        msg = f"parameter {param!r} contains an empty list item"
        raise UsageError(msg)
    return parts


def _parse_dict(raw: str, param: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"parameter {param!r} expects a JSON object, got {raw!r}"
        raise UsageError(msg) from exc
    if not isinstance(value, dict):
        msg = f"parameter {param!r} expects a JSON object, got {raw!r}"
        raise UsageError(msg)
    return cast(dict[str, Any], value)


def _settable_fields(cls: type[Any]) -> dict[str, _Field]:
    fields: dict[str, _Field] = {}
    for name, _param, annotation in _init_parameters(cls):
        field = _field_for(annotation)
        if field is not None:
            fields[name] = field
    return fields


def _blocked_params(cls: type[Any]) -> set[str]:
    return {
        name for name, _param, annotation in _init_parameters(cls) if _field_for(annotation) is None
    }


def _init_parameters(cls: type[Any]) -> list[tuple[str, inspect.Parameter, Any]]:
    init = cast(Callable[..., None], cls.__init__)  # type: ignore[misc]
    signature = inspect.signature(init)
    try:
        hints = get_type_hints(init)
    except Exception:
        hints = {}
    rows: list[tuple[str, inspect.Parameter, Any]] = []
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        rows.append((name, param, hints.get(name, param.annotation)))
    return rows


def _plugin_label(group: str, name: str, cls: type[Any]) -> str:
    if group == "explainers" and name == "llm":
        return "llm (needs --provider)"
    required = [param for param in required_param_names(cls) if param in _settable_fields(cls)]
    if not required:
        return name
    return f"{name} (needs {', '.join(required)})"


def _field_for(annotation: Any) -> _Field | None:
    optional = _contains_none(annotation)
    if _is_str_list(annotation) or _union_has_str_list(annotation):
        return _Field("str_list", optional)
    if _is_dict(annotation) or _union_has_dict(annotation):
        return _Field("dict", optional)
    scalar = _unwrap_scalar(annotation)
    if scalar is bool:
        return _Field("bool", optional)
    if scalar is int:
        return _Field("int", optional)
    if scalar is float:
        return _Field("float", optional)
    if scalar is str:
        return _Field("str", optional)
    return None


def _contains_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        return any(arg is type(None) for arg in get_args(annotation))
    return False


def _is_str_list(annotation: Any) -> bool:
    if get_origin(annotation) is not list:
        return False
    args = get_args(annotation)
    return len(args) == 1 and args[0] is str


def _is_dict(annotation: Any) -> bool:
    return get_origin(annotation) is dict


def _union_has_str_list(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return False
    return any(_is_str_list(arg) for arg in get_args(annotation) if arg is not type(None))


def _union_has_dict(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return False
    return any(_is_dict(arg) for arg in get_args(annotation) if arg is not type(None))


def _unwrap_scalar(annotation: Any) -> type[Any] | None:
    if annotation in _SCALAR_TYPES:
        return cast(type[Any], annotation)
    if isinstance(annotation, str):
        return {"str": str, "int": int, "float": float, "bool": bool}.get(annotation)
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _unwrap_scalar(args[0])
        return None
    if origin is Literal:
        values = get_args(annotation)
        if not values:
            return None
        if all(isinstance(value, str) for value in values):
            return str
        if all(isinstance(value, bool) for value in values):
            return bool
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            return int
    return None
