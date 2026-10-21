"""The Python API reference (docs/API.md, docs/api/*.md) is generated from the docstrings and stays complete."""

import contextlib
import importlib
import inspect
import re
import shutil
from pathlib import Path

import pytest

from amgeo_sd import api_doc

ROOT = Path(__file__).parent.parent
SUBPACKAGES = ["models", "io", "filters", "clustering", "scatter", "geo", "plotting", "pipelines", "export", "config"]


def test_generated_api_files_are_up_to_date():
    for path, text in api_doc.generate(ROOT / "docs").items():
        assert path.read_text() == text, f"{path.relative_to(ROOT)} is stale: run `make docs`"


def test_no_stray_files_in_docs_api():
    expected = {p.name for p in api_doc.generate(ROOT / "docs") if p.parent.name == "api"}
    assert {p.name for p in (ROOT / "docs" / "api").glob("*.md")} == expected


def test_every_parameter_of_every_public_function_is_documented():
    assert api_doc.problems() == []


def test_every_public_name_is_on_a_page():
    covered = {(p.module, n) for p in api_doc.PAGES for n in api_doc.public_names(p.module, p.names)}
    for name in SUBPACKAGES:
        module = f"amgeo_sd.{name}"
        for exported in importlib.import_module(module).__all__:
            assert (module, exported) in covered, f"{module}.{exported} is exported but not in api_doc.PAGES"


def test_public_exports_exist_and_are_defined_in_the_package():
    for page in api_doc.PAGES:
        mod = importlib.import_module(page.module)
        for name in api_doc.public_names(page.module, page.names):
            obj = getattr(mod, name)
            if inspect.isclass(obj) or inspect.isfunction(obj):
                assert obj.__module__.startswith("amgeo_sd"), f"{page.module}.{name}"


def test_docstring_parser_reads_parameters_returns_and_raises():
    doc = api_doc.parse_docstring(
        """Summary line
        continues.

        More text.

        Parameters
        ----------
        a, b : int
            First and
            second.
        c : str, optional
            Third.

        Returns
        -------
        float
            The result.

        Raises
        ------
        ValueError
            When wrong.
        """
    )
    assert doc.summary == "Summary line continues."
    assert doc.body == "More text."
    assert [(e.names, e.type) for e in doc.sections["Parameters"]] == [(["a", "b"], "int"), (["c"], "str, optional")]
    assert doc.sections["Parameters"][0].text == "First and second."
    assert doc.sections["Returns"][0].type == "float" and doc.sections["Returns"][0].text == "The result."
    assert doc.sections["Raises"][0].type == "ValueError"


def test_checker_flags_missing_stale_and_untyped_parameters():
    def f(x, y=1):
        """Do it.

        Parameters
        ----------
        x : int
            Documented.
        z : int
            Not in the signature.
        """

    out: list[str] = []
    api_doc._check_function("f", f, api_doc.parse_docstring(inspect.getdoc(f)), out)
    assert any("`y` is not documented" in m for m in out)
    assert any("`z` is not in the signature" in m for m in out)


def test_roles_become_links_or_code():
    text = api_doc._md("see :func:`amgeo_sd.io.load_frame`, :class:`~amgeo_sd.models.Scan` and :func:`nowhere.unknown`")
    assert "[`load_frame`](io.md#load_frame)" in text
    assert "[`Scan`](models.md#scan)" in text
    assert text.endswith("`unknown`")
    assert ":func:" not in text


def test_api_pages_link_to_each_other_and_from_the_main_documents():
    def links(name):
        return set(re.findall(r"\]\(([^)\s#]+)", (ROOT / name).read_text()))

    assert "docs/API.md" in links("README.md")
    assert "API.md" in links("docs/USAGE.md") and "API.md" in links("docs/README.md")
    assert "API.md" in links("docs/PARAMETERS.md")
    for page in api_doc.PAGES:
        assert f"api/{page.slug}.md" in links("docs/API.md")


@pytest.mark.realdata
@pytest.mark.parametrize("title", list(api_doc.WORKFLOWS))
def test_workflow_examples_run_on_real_data(title, fitacf_files, monkeypatch):
    """The snippets printed in docs/API.md are executed, so the documented calls cannot rot."""
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.chdir(ROOT)
    try:
        exec(compile(api_doc.WORKFLOWS[title], title, "exec"), {"__name__": "workflow"})
    finally:
        shutil.rmtree(ROOT / "data" / "outputs" / "api-example", ignore_errors=True)
        with contextlib.suppress(Exception):
            import matplotlib.pyplot as plt

            plt.close("all")
