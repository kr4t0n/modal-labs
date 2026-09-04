"""Repository-level invariants that no single project's suite would catch.

The one that matters: pytest only collects what `testpaths` lists, so adding a
service directory without adding its suite means those tests never run. Nothing
fails — `pytest -q` stays green and simply says less than it appears to. This
file is what notices.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Not built or tested by design; see deprecated/README.md.
EXCLUDED = {"deprecated"}


def configured_testpaths() -> set[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(config["tool"]["pytest"]["ini_options"]["testpaths"])


def discovered_suites() -> set[str]:
    """Every `<project>/tests` directory that holds at least one test module."""
    found = set()
    for candidate in ROOT.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        if candidate.name in EXCLUDED:
            continue
        suite = candidate / "tests"
        if suite.is_dir() and any(suite.glob("test_*.py")):
            found.add(f"{candidate.name}/tests")
    return found


def test_every_project_suite_is_collected():
    missing = discovered_suites() - configured_testpaths()
    assert not missing, (
        f"these suites exist but pytest never runs them: {sorted(missing)}. "
        "Add them to [tool.pytest.ini_options] testpaths in pyproject.toml."
    )


def test_no_configured_testpath_is_stale():
    """A renamed project leaves a path that silently collects nothing."""
    stale = {path for path in configured_testpaths() if not (ROOT / path).is_dir()}
    assert not stale, f"testpaths entries pointing at nothing: {sorted(stale)}"


def test_every_service_ships_the_expected_modules():
    """A service is `app.py` + `server.py` + `workflow.py` + docs, by convention."""
    for suite in sorted(discovered_suites()):
        project = ROOT / suite.split("/")[0]
        for required in ("app.py", "server.py", "workflow.py", "README.md", "AGENTS.md"):
            assert (project / required).is_file(), f"{project.name} is missing {required}"


def test_every_service_is_in_the_ci_import_check():
    """CI imports each app.py in its own process; a new one must be added there.

    Same failure shape as testpaths: the check passes while covering less.
    """
    workflow_text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    line = next(line for line in workflow_text.splitlines() if "for project in" in line)
    listed = set(line.split("for project in")[1].split(";")[0].split())
    projects = {suite.split("/")[0] for suite in discovered_suites()}
    assert projects <= listed, f"missing from the CI import check: {sorted(projects - listed)}"
