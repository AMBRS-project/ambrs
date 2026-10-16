"""Regression tests for working-directory-independent tutorial path lookup."""

import ast
import json
from pathlib import Path

import pytest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "tutorial/notebooks/02_run_prepare_plot/01_single_scenario_partmc.ipynb"
)


def _find_tutorial_root_function():
    """Load the real function from the notebook without running PartMC."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def find_tutorial_root(" not in source:
            continue
        module = ast.parse(source)
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "find_tutorial_root"
        )
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), scope)
        return scope["find_tutorial_root"]
    raise AssertionError("Notebook does not define find_tutorial_root")


def _make_tutorial(tmp_path):
    repo = tmp_path / "ambrs"
    (repo / "tutorial/notebooks/02_run_prepare_plot").mkdir(parents=True)
    return repo


def test_find_tutorial_root_from_repository_root(tmp_path, monkeypatch):
    repo = _make_tutorial(tmp_path)
    monkeypatch.chdir(repo)
    assert _find_tutorial_root_function()() == repo / "tutorial"


def test_find_tutorial_root_from_notebook_directory(tmp_path, monkeypatch):
    repo = _make_tutorial(tmp_path)
    monkeypatch.chdir(repo / "tutorial/notebooks/02_run_prepare_plot")
    assert _find_tutorial_root_function()() == repo / "tutorial"


def test_find_tutorial_root_from_explicit_start(tmp_path):
    repo = _make_tutorial(tmp_path)
    nested = repo / "other/nested"
    nested.mkdir(parents=True)
    assert _find_tutorial_root_function()(str(nested)) == repo / "tutorial"
    assert _find_tutorial_root_function()(nested) == repo / "tutorial"


def test_find_tutorial_root_errors_outside_repository(tmp_path, monkeypatch):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    with pytest.raises(RuntimeError, match="Could not locate the AMBRS tutorial directory"):
        _find_tutorial_root_function()()


def test_find_tutorial_root_requires_notebook_directory(tmp_path, monkeypatch):
    repo = tmp_path / "ambrs"
    (repo / "tutorial").mkdir(parents=True)
    monkeypatch.chdir(repo)
    with pytest.raises(RuntimeError, match="Could not locate the AMBRS tutorial directory"):
        _find_tutorial_root_function()()
