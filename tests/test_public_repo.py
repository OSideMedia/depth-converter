"""What a stranger reads and runs on the public repo holds true.

The badges are claims and the README order is the front door (a reader who cannot find out what
this is, or whether it needs OSIDE, closes the tab). The rest is the hardening a stranger relies on
without ever reading it: what setup.sh downloads, how the checkpoint loads, what the UI listens on.
Each predicate is run on the counterexample it exists to refuse, so none of them is only a claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
LAUNCHERS = (ROOT / "scripts" / "build_launchers.sh").read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------- setup and the lock

def test_setup_stops_on_a_macos_the_locked_torch_has_no_wheel_for():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    wheels = re.findall(r"torch-[0-9.]+-[^\s\"]*macosx_(\d+)_0_arm64\.whl", lock)
    assert wheels, "no torch wheels for this platform in the lock: this check read nothing"
    oldest = min(int(major) for major in wheels)
    assert f'-lt {oldest} ]' in SETUP
    assert SETUP.index("sw_vers -productVersion") < SETUP.index("install.sh")
    assert f"macOS {oldest} or newer" in README
    assert f"<string>{oldest}.0</string>" in LAUNCHERS


def test_setup_puts_no_python_on_the_path():
    assert re.search(r'python install --no-bin cpython-', SETUP)
    assert not re.search(r'python install cpython-', SETUP)


# ---------------------------------------------------------------------- the launchers

def _build_launchers(root: Path, apps: Path, home: Path, port: str = "7861") -> subprocess.CompletedProcess:
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "build_launchers.sh", root / "scripts" / "build_launchers.sh")
    return subprocess.run(
        ["bash", str(root / "scripts" / "build_launchers.sh"), str(apps)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "HOME": str(home), "DEPTHC_PORT": port},
    )


def test_a_project_folder_name_is_data_in_the_launcher(tmp_path):
    root = tmp_path / "repo$(touch PWNED)"
    built = _build_launchers(root, tmp_path, tmp_path)
    assert built.returncode == 0, built.stderr
    script = (tmp_path / "Depth Converter.app" / "Contents" / "MacOS" / "run").read_text()
    assignment = next(line for line in script.splitlines() if line.startswith("ROOT="))
    read_back = subprocess.run(
        ["bash", "-c", f'{assignment}\nprintf %s "$ROOT"'], cwd=tmp_path, capture_output=True, text=True,
    )
    assert read_back.stdout == str(root)
    assert not (tmp_path / "PWNED").exists()


def test_a_port_that_is_not_a_number_stops_the_build(tmp_path):
    built = _build_launchers(tmp_path / "repo", tmp_path, tmp_path, port="1;touch PWNED")
    assert built.returncode == 1
    assert not (tmp_path / "Depth Converter.app").exists()


def _a_process(argv0: str, cwd: Path) -> subprocess.Popen:
    # exec -a sets what `ps -o command=` shows, one of the two things the Stop app reads;
    # the other is the folder the process runs in
    cwd.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["bash", "-c", f'exec -a "{argv0}" sleep 60'], cwd=cwd)


def _run_stop_with_pidfile_naming(process: subprocess.Popen, tmp_path: Path) -> None:
    built = _build_launchers(tmp_path / "repo", tmp_path, tmp_path, port="47911")
    assert built.returncode == 0, built.stderr
    (tmp_path / ".depthc-ui.pid").write_text(str(process.pid))
    subprocess.run(
        ["bash", str(tmp_path / "Depth Stop.app" / "Contents" / "MacOS" / "run")],
        check=True, timeout=30, env={**os.environ, "HOME": str(tmp_path)},
    )
    assert not (tmp_path / ".depthc-ui.pid").exists()


@pytest.mark.parametrize("argv0", ["someone-elses-server", "python3 -m http.server --directory /tmp/depthc ui"])
def test_stop_leaves_a_process_that_is_not_ours(tmp_path, argv0):
    # The second one runs elsewhere with our words in its arguments: the words alone prove nothing.
    other = _a_process(argv0, tmp_path / "elsewhere")
    try:
        _run_stop_with_pidfile_naming(other, tmp_path)
        assert other.poll() is None, "Stop killed a process that was not the app"
    finally:
        other.kill()
        other.wait()


def test_stop_stops_ours(tmp_path):
    ours = _a_process("depthc ui --port 47911", tmp_path / "repo")
    try:
        _run_stop_with_pidfile_naming(ours, tmp_path)
        assert ours.wait(timeout=10) is not None
    finally:
        ours.kill()
        ours.wait()


def test_stop_and_start_look_only_at_listeners():
    assert LAUNCHERS.count("-sTCP:LISTEN") == 3
    assert "lsof -ti :" not in LAUNCHERS  # that form also matches a browser tab using the port


# ---------------------------------------------------------------------- what gets pasted

def test_the_paste_lines_quote_what_they_echo():
    comparison = SimpleNamespace(ratio=1.05, r0=0.9, suggestion="adhered", overlap_seconds=5.0)
    hint = cli._record_hint("kit$(touch PWNED)", comparison, None)
    assert "'kit$(touch PWNED)'" in hint
    source = (ROOT / "depthc" / "cli.py").read_text(encoding="utf-8")
    assert "depthc kit {shlex.quote(str(clip))}" in source


# ---------------------------------------------------------------------- the words

def test_the_2_5_advice_does_not_call_a_measured_tier_unmeasured():
    source = (ROOT / "depthc" / "kit.py").read_text(encoding="utf-8") \
        + (ROOT / "depthc" / "templates" / "seedance-2.5.txt").read_text(encoding="utf-8")
    assert "UNMEASURED" not in source


def test_the_readme_does_not_promise_every_ref_is_720p():
    assert "always ≤720p" not in README
    assert "cannot fail if the code is" not in README


def test_no_tracked_doc_carries_a_ticket_id_a_reader_cannot_look_up():
    ticket = re.compile(r"\bow-[0-9a-f]{6}\b")
    assert ticket.search("(ow-4ccd14)")
    readers = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "DOCS").glob("*.md")),
               *sorted((ROOT / "depthc").glob("*.py"))]
    assert [p.name for p in readers if ticket.search(p.read_text(encoding="utf-8"))] == []
