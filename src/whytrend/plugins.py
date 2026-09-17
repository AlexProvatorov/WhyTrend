"""Discover and load WhyTrend plugins via packaging entry points."""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Final

PLUGIN_GROUPS: Final[tuple[str, ...]] = (
    "whytrend.sources",
    "whytrend.detectors",
    "whytrend.collectors",
    "whytrend.rankers",
    "whytrend.explainers",
)

_GROUP_ALIASES: Final[dict[str, str]] = {
    "sources": "whytrend.sources",
    "detectors": "whytrend.detectors",
    "collectors": "whytrend.collectors",
    "rankers": "whytrend.rankers",
    "explainers": "whytrend.explainers",
}


class PluginError(Exception):
    """Raised when a plugin group or entry point cannot be resolved."""


class UnknownPluginError(PluginError):
    """Raised when the requested plugin name is not registered."""


def list_plugins(group: str) -> dict[str, type]:
    """Return ``{name: class}`` for all entry points in ``group``."""
    normalized = _normalize_group(group)
    plugins: dict[str, type] = {}
    for point in _iter_entry_points(normalized):
        plugins[point.name] = _load_entry_point(point)
    return plugins


def load_plugin(group: str, name: str) -> type:
    """Load a single plugin class by ``group`` and ``name``."""
    plugins = list_plugins(group)
    try:
        return plugins[name]
    except KeyError:
        available = ", ".join(sorted(plugins)) or "(none)"
        msg = (
            f"unknown plugin {name!r} in group {_normalize_group(group)!r}; available: {available}"
        )
        raise UnknownPluginError(msg) from None


def _normalize_group(group: str) -> str:
    if group in PLUGIN_GROUPS:
        return group
    aliased = _GROUP_ALIASES.get(group)
    if aliased is not None:
        return aliased
    expected = ", ".join(PLUGIN_GROUPS)
    msg = f"unknown plugin group {group!r}; expected one of: {expected}"
    raise PluginError(msg)


def _iter_entry_points(group: str) -> tuple[EntryPoint, ...]:
    return tuple(entry_points(group=group))


def _load_entry_point(point: EntryPoint) -> type:
    try:
        loaded = point.load()
    except Exception as exc:
        msg = f"failed to load plugin {point.name!r} from group {point.group!r}"
        raise PluginError(msg) from exc
    if not isinstance(loaded, type):
        msg = f"plugin {point.name!r} did not resolve to a class"
        raise PluginError(msg)
    return loaded
