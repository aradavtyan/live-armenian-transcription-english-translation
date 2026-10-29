import importlib

import pytest

import live_translate

MODULES = ["audio.capture", "cli", "core", "models", "pipeline"]


def test_version_is_set():
    assert live_translate.__version__


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"live_translate.{name}")
