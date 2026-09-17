from __future__ import annotations

from importlib.metadata import EntryPoint
from unittest.mock import patch

import pytest

from tests.plugin_support import FakeNewsCollector, FakeSpikeDetector
from whytrend.collectors import HackerNewsCollector
from whytrend.detectors import ZScoreDetector
from whytrend.plugins import (
    PLUGIN_GROUPS,
    PluginError,
    UnknownPluginError,
    list_plugins,
    load_plugin,
)


def _entry(name: str, value: str, group: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


def _fake_entry_points(group: str | None = None, **kwargs):
    _ = kwargs
    detectors = (
        _entry(
            "fake_spike",
            "tests.plugin_support:FakeSpikeDetector",
            "whytrend.detectors",
        ),
    )
    collectors = (
        _entry(
            "fake_news",
            "tests.plugin_support:FakeNewsCollector",
            "whytrend.collectors",
        ),
    )
    mapping = {
        "whytrend.detectors": detectors,
        "whytrend.collectors": collectors,
    }
    if group is None:
        return detectors + collectors
    return mapping.get(group, ())


def test_list_and_load_fake_detector_and_collector() -> None:
    with patch("whytrend.plugins.entry_points", side_effect=_fake_entry_points):
        detectors = list_plugins("whytrend.detectors")
        collectors = list_plugins("collectors")

        assert detectors["fake_spike"] is FakeSpikeDetector
        assert collectors["fake_news"] is FakeNewsCollector
        assert load_plugin("whytrend.detectors", "fake_spike") is FakeSpikeDetector
        assert load_plugin("whytrend.collectors", "fake_news") is FakeNewsCollector


def test_unknown_plugin_name_lists_available() -> None:
    with (
        patch("whytrend.plugins.entry_points", side_effect=_fake_entry_points),
        pytest.raises(UnknownPluginError, match="unknown plugin 'nope'") as exc_info,
    ):
        load_plugin("whytrend.detectors", "nope")

    message = str(exc_info.value)
    assert "whytrend.detectors" in message
    assert "fake_spike" in message


def test_unknown_plugin_group() -> None:
    with pytest.raises(PluginError, match="unknown plugin group"):
        list_plugins("whytrend.widgets")


def test_plugin_must_resolve_to_a_class() -> None:
    point = _entry(
        "not_a_class", "tests.plugin_support:FakeSpikeDetector.name", "whytrend.detectors"
    )

    def _entries(group: str | None = None, **kwargs):
        _ = kwargs
        if group == "whytrend.detectors":
            return (point,)
        return ()

    with (
        patch("whytrend.plugins.entry_points", side_effect=_entries),
        pytest.raises(PluginError, match="did not resolve to a class"),
    ):
        load_plugin("whytrend.detectors", "not_a_class")


def test_builtin_entry_points_if_package_installed() -> None:
    detectors = list_plugins("whytrend.detectors")
    collectors = list_plugins("whytrend.collectors")
    if "zscore" not in detectors or "hacker_news" not in collectors:
        pytest.skip("whytrend entry points are not installed in this environment")

    assert load_plugin("whytrend.detectors", "zscore") is ZScoreDetector
    assert load_plugin("whytrend.collectors", "hacker_news") is HackerNewsCollector
    assert set(PLUGIN_GROUPS) >= {
        "whytrend.detectors",
        "whytrend.collectors",
    }


def test_direct_imports_still_work() -> None:
    assert ZScoreDetector().name == "zscore"
    assert HackerNewsCollector().name == "hacker_news"
