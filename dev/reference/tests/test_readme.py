"""The README must describe commands and files that exist."""
import re
import tomllib
from pathlib import Path

import yaml

REFERENCE = Path(__file__).resolve().parents[1]
README = (REFERENCE / "README.md").read_text()
MANIFEST = tomllib.loads((REFERENCE / "pixi.toml").read_text())
CONFIG = yaml.safe_load((REFERENCE / "config.yaml").read_text())


def defined_tasks() -> set[str]:
    tasks = set(MANIFEST.get("tasks", {}))
    for feature in MANIFEST.get("feature", {}).values():
        tasks |= set(feature.get("tasks", {}))
    return tasks


def readme_task_names() -> set[str]:
    """Task names in every ``pixi run ...`` of the README (raw ``python`` commands excluded)."""
    names = set()
    for command in re.findall(r"pixi run ([^\n`]*)", README):
        tokens = command.split()
        while tokens and tokens[0].startswith("-"):  # options such as -e mantid615
            flag = tokens.pop(0)
            if flag in ("-e", "--environment", "--manifest-path"):
                tokens.pop(0)
        if tokens and tokens[0] not in ("python", "..."):
            names.add(tokens[0])
    return names


def test_readme_commands_exist():
    tasks = defined_tasks()
    used = readme_task_names()
    assert used, "no pixi commands found in the README"
    assert used <= tasks, f"README mentions undefined tasks: {sorted(used - tasks)}"


def test_every_task_is_documented():
    undocumented = {t for t in defined_tasks() if f"`{t}`" not in README and f"pixi run {t}" not in README}
    # Helper tasks used only by composite ones
    assert undocumented <= {"clean-build", "size", "compare-report", "generate-mantid611"}, undocumented


def test_environment_table_matches_config():
    """The table of environments -> reference files lists exactly the configured files."""
    table = {}
    for line in README.splitlines():
        match = re.match(r"\| `(mantid\d+)` \| [\d.]+ \| [\d.]+ \| (.*) \|$", line)
        if match:
            table[match[1]] = set(re.findall(r"`([^`]+)`", match[2]))

    expected = {}
    for entry in CONFIG["pychop"]:
        expected.setdefault(entry["environment"], set()).add(
            f"pychop/{entry['instrument']}_{entry['model_version']}"
        )
    for name in ("pychop", "panther", "tosca", "lagrange"):
        expected.setdefault(CONFIG["abins"]["environment"], set()).add(f"abins/{name}")

    assert table == expected


def test_readme_environments_exist():
    environments = set(MANIFEST["environments"])
    for name in re.findall(r"-e (mantid\d+)", README):
        # mantid616 is the hypothetical environment of the "new release" example
        assert name in environments or name == "mantid616"
