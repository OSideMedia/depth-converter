"""What a stranger reads and runs on the public repo holds true.

The badges are claims and the README order is the front door (a reader who cannot find out what
this is, or whether it needs OSIDE, closes the tab). The rest is the hardening a stranger relies on
without ever reading it: what setup.sh downloads, how the checkpoint loads, what the UI listens on.
Each predicate is run on the counterexample it exists to refuse, so none of them is only a claim.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import depthc
from depthc import cli, ui, vendor_path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
SETUP = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")


# ---------------------------------------------------------------------- the badges

def test_the_version_badge_pyproject_and_package_agree():
    badge = re.search(r"img\.shields\.io/badge/version-([0-9][^-\s)]*)-", README)
    assert badge, "the README has no version badge: this gate has no subject"
    assert badge.group(1) == PYPROJECT["project"]["version"] == depthc.__version__


def test_the_licence_badge_file_and_metadata_agree():
    assert "badge/license-MIT" in README
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert PYPROJECT["project"]["license"] == "MIT"
    for pattern in PYPROJECT["project"]["license-files"]:
        assert list(ROOT.glob(pattern)), f"license-files names {pattern}, which matches nothing"


# ---------------------------------------------------------------------- the front door

def _heading_lines() -> dict[str, int]:
    return {line: n for n, line in enumerate(README.splitlines()) if line.startswith("## ")}


def test_the_readme_answers_in_the_order_a_new_reader_asks():
    lines = _heading_lines()
    order = ["## What it does", "## Do I need OSIDE to use this?", "## Requirements", "## Install",
             "## Quick start"]
    assert [lines[h] for h in order] == sorted(lines[h] for h in order)


def test_install_sits_in_the_first_quarter():
    total = len(README.splitlines())
    assert _heading_lines()["## Install"] < total / 4


def test_the_answer_about_oside_comes_with_a_minimal_input():
    section = README.split("## Do I need OSIDE to use this?")[1].split("\n## ")[0]
    assert section.lstrip().startswith("**No.**")
    assert "depthc kit my-clip.mp4" in section


# Plan phases, ticket ids and audit tags mean nothing outside this repo's history, and wording that
# offers a way past a provider's rules does not belong on the front door either.
BANNED = re.compile(
    r"Phase \d|Gate-\d|PLAN-SHARED|LIVEFIRE-\d|audit [A-Z]+-\d|\bow-[0-9a-f]{6}\b|"
    r"sidestep|bypass|probing (with|the)|moderation boundary",
    re.I,
)


def test_the_readme_carries_no_internal_jargon_or_filter_talk():
    hit = BANNED.search(README)
    assert hit is None, repr(hit.group(0))


def test_the_banned_pattern_can_fire():
    for old in ("the Phase 4 console check", "the exact Gate-4 measurement", "(audit DC-6)",
                "sidesteps the real-face input filter", "so probing with this is cheap",
                "Built for the moderation boundary", "(ow-4ccd14)"):
        assert BANNED.search(old), old


# ---------------------------------------------------------------------- setup.sh

def _verify_ckpt_result(path: Path, sha: str) -> int:
    body = re.search(r"^verify_ckpt\(\) \{\n.*?^\}\n", SETUP, re.M | re.S)
    assert body, "setup.sh has no verify_ckpt function: this gate has no subject"
    script = f'{body.group(0)}\nCKPT_SHA256="{sha}"\nverify_ckpt "$1"\n'
    return subprocess.run(["bash", "-c", script, "verify", str(path)]).returncode


def test_verify_ckpt_refuses_the_wrong_bytes(tmp_path):
    good = tmp_path / "ckpt.pth"
    good.write_bytes(b"the real weights")
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    assert _verify_ckpt_result(good, sha) == 0
    good.write_bytes(b"the real weightz")
    assert _verify_ckpt_result(good, sha) != 0


def test_the_download_is_verified_before_it_lands():
    fetch = SETUP.index('-o "$CKPT.part"')
    check = SETUP.index('verify_ckpt "$CKPT.part"')
    land = SETUP.index('mv "$CKPT.part" "$CKPT"')
    assert fetch < check < land


@pytest.mark.slow
def test_the_pinned_hash_is_the_checkpoint_the_model_loads():
    pinned = re.search(r'^CKPT_SHA256="([0-9a-f]{64})"$', SETUP, re.M)
    assert pinned
    assert hashlib.sha256(vendor_path.CHECKPOINT.read_bytes()).hexdigest() == pinned.group(1)


# The variable has to sit on `sh`, the installer: on `curl` it never reaches it, and the installer
# writes to the home folder and edits the shell profile.
UV_INSTALL = re.compile(r'curl [^\n|]*uv/\$\{UV_VERSION\}/install\.sh" \| UV_UNMANAGED_INSTALL="[^"\n]+" sh$', re.M)


def test_uv_is_pinned_and_installed_into_the_project():
    assert UV_INSTALL.search(SETUP), "setup.sh does not install the pinned uv into .tools/"
    assert re.search(r'^UV_VERSION="\d+\.\d+\.\d+"$', SETUP, re.M)
    assert not UV_INSTALL.search(
        'UV_UNMANAGED_INSTALL="$ROOT/.tools" curl -LsSf https://astral.sh/uv/install.sh | sh\n'
    )


def test_setup_stops_early_where_it_cannot_run():
    guard = SETUP.index('"$(uname -m)" != "arm64"')
    assert guard < SETUP.index("install.sh"), "the platform check must come before any download"


# ---------------------------------------------------------------------- what it runs with

def _unsafe_torch_loads(source: str) -> list[str]:
    calls = re.findall(r"torch\.load\([^\n]*", source)
    return [call for call in calls if "weights_only=True" not in call]


def test_every_torch_load_refuses_pickled_code():
    subjects = sorted((ROOT / "depthc").glob("**/*.py"))
    loads = [path for path in subjects if "torch.load(" in path.read_text(encoding="utf-8")]
    assert len(loads) >= 2, "the sweep found fewer torch.load sites than engine.py and smoke.py"
    for path in loads:
        assert _unsafe_torch_loads(path.read_text(encoding="utf-8")) == [], path.name
    assert _unsafe_torch_loads('model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)')


def test_the_ci_token_is_read_only():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert re.search(r"^permissions:\n  contents: read$", workflow, re.M)


def test_the_ui_listens_on_loopback_only(monkeypatch):
    seen: dict = {}

    class Fake:
        def launch(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(ui, "build", Fake)
    ui.launch(open_browser=False)
    assert seen["server_name"] == "127.0.0.1"
    assert seen["share"] is False


def test_the_cli_offers_no_public_share():
    help_text = CliRunner().invoke(cli.app, ["ui", "--help"]).output
    assert "--port" in help_text, "the help did not render: this gate has no subject"
    assert "share" not in help_text.lower()


def test_the_compare_paste_line_assumes_no_surface_without_an_oside_id():
    comparison = SimpleNamespace(ratio=1.05, r0=0.9, suggestion="adhered", overlap_seconds=5.0)
    assert "--surface" not in cli._record_hint("Kit_001", comparison, None)
    with_id = cli._record_hint("Kit_001", comparison, "3f2504e0-4f89-4d3c-9a6b-0305e82c3301")
    assert "--surface oside --generation 3f2504e0-4f89-4d3c-9a6b-0305e82c3301" in with_id


def test_gradio_analytics_are_off():
    assert ui.build().analytics_enabled is False


def test_the_schema_ids_point_at_files_in_this_repo():
    base = "https://raw.githubusercontent.com/OSideMedia/depth-converter/main/"
    for name in ("kit_schema.json", "result_schema.json"):
        schema_id = json.loads((ROOT / "depthc" / name).read_text(encoding="utf-8"))["$id"]
        assert schema_id.startswith(base), schema_id
        assert (ROOT / schema_id.removeprefix(base)).is_file(), schema_id
