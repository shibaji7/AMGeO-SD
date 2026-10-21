"""The documentation stays consistent with the code: generated parameter reference, flags, make variables, links, file names."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from amgeo_sd.cli import build_parser
from amgeo_sd.config import FIELD_DOCS, Settings
from amgeo_sd.params_doc import MAKE_VARIABLES, PARAM_META, parameters_markdown

ROOT = Path(__file__).parent.parent
MARKDOWN = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "docs" / "api").glob("*.md")),
    ROOT / "configs" / "README.md",
    ROOT / "tests" / "README.md",
    ROOT / "tests" / "data" / "README.md",
    ROOT / "_bkp" / "README.md",
    ROOT / "examples" / "median_day_analysis" / "README.md",
]
#: which command-line sub-commands read each settings table
COMMANDS = {
    "run": ["median", "dbscan"],
    "input": ["median", "dbscan"],
    "median": ["median"],
    "dbscan": ["dbscan"],
    "geometry": ["median", "dbscan", "plot"],
    "output": ["median", "dbscan"],
    "plot": ["plot"],
}


def settings_keys():
    s = Settings()
    return [(table, f.name) for table in s.to_dict() for f in fields(getattr(s, table))]


def test_parameters_md_is_up_to_date():
    assert (ROOT / "docs" / "PARAMETERS.md").read_text() == parameters_markdown(), "run `make docs`"


def test_every_setting_is_described_and_nothing_stale():
    keys = settings_keys()
    assert set(PARAM_META) == set(keys)
    for table, key in keys:
        assert FIELD_DOCS[table].get(key), f"[{table}] {key} has no description"
        text = parameters_markdown()
        assert re.search(rf"^\| `{key}` \|", text, re.M), (table, key)


def test_command_line_flags_in_the_reference_exist():
    sub = next(a for a in build_parser()._subparsers._group_actions).choices
    for (table, key), (_, cli, _) in PARAM_META.items():
        if cli == "-":
            continue
        for flag in re.findall(r"`(-[-\w]+)`", cli):
            assert any(flag in sub[cmd]._option_string_actions for cmd in COMMANDS[table]), f"[{table}] {key}: {flag}"


def test_make_variables_match_the_makefile():
    defined = set(re.findall(r"^([A-Z_]+)\s*\?=", (ROOT / "Makefile").read_text(), re.M))
    assert defined == set(MAKE_VARIABLES), (
        f"undocumented {defined - set(MAKE_VARIABLES)}, stale {set(MAKE_VARIABLES) - defined}"
    )
    for (table, key), (_, _, make) in PARAM_META.items():
        for var in re.findall(r"`([A-Z_]+)`", make):
            assert var in MAKE_VARIABLES, f"[{table}] {key}: {var}"


def test_default_toml_lists_every_key():
    text = (ROOT / "configs" / "default.toml").read_text()
    tables = re.split(r"^\[(\w+)\]\s*$", text, flags=re.M)  # ['', name, body, name, body, ...]
    body = dict(zip(tables[1::2], tables[2::2], strict=True))
    for table, key in settings_keys():
        assert re.search(rf"^#?\s*{key}\s*=", body[table], re.M), f"configs/default.toml [{table}] lacks {key}"


def slug(heading: str) -> str:
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


def anchors(path: Path) -> set[str]:
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", path.read_text(), re.M)}


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path):
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)  # code blocks hold examples, not links
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file, _, anchor = target.partition("#")
        dest = path if not file else (path.parent / file).resolve()
        assert dest.exists(), f"{path.relative_to(ROOT)}: broken link {target}"
        if anchor and dest.suffix == ".md":
            assert anchor in anchors(dest), f"{path.relative_to(ROOT)}: no heading for #{anchor} in {dest.name}"


def test_main_documents_link_to_each_other():
    def links(name):
        return set(re.findall(r"\]\(([^)\s#]+)", (ROOT / name).read_text()))

    assert {"docs/USAGE.md", "docs/PARAMETERS.md", "docs/PORTING.md", "docs/README.md"} <= links("README.md")
    assert {"PARAMETERS.md", "../README.md"} <= links("docs/USAGE.md")
    assert {"USAGE.md", "../README.md", "../configs/README.md", "../tests/README.md", "../_bkp/README.md"} <= links(
        "docs/PARAMETERS.md"
    )
    assert {"USAGE.md", "PARAMETERS.md", "PORTING.md"} <= links("docs/README.md")


def test_docs_use_the_current_file_names():
    for path in MARKDOWN + [ROOT / "Makefile", ROOT / "configs" / "default.toml"]:
        text = path.read_text()
        assert ".fitacf++.nc" not in text, f"{path.name} still uses the old NetCDF name"
        assert "fitacf.pl.bz2" not in text, f"{path.name} mentions the old .pl file name"
    usage = (ROOT / "docs" / "USAGE.md").read_text()
    assert ".pp.nc" in usage and ".fitacf.pp.bz2" in usage


def test_extra_fields_are_documented():
    from amgeo_sd.io import EXTRA_RECORD_FIELDS

    usage = (ROOT / "docs" / "USAGE.md").read_text()
    section = usage.split("### The extra fields")[1].split("\n### ")[0]
    for field in EXTRA_RECORD_FIELDS:
        assert f"| `{field}` |" in section, f"{field} is not in the extra-fields table of docs/USAGE.md"
    for text in ("README.md", "docs/PARAMETERS.md", "configs/default.toml"):
        assert "extra_fields" in (ROOT / text).read_text() or "extra-fields" in (ROOT / text).read_text(), text
    assert "#the-extra-fields" in (ROOT / "README.md").read_text()
