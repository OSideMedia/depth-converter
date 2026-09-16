"""One handover truth across every surface that states it (audit DC-2).

The verified fact (live run 2026-07-25): OSIDE's Composer has NO binding field for an
uploaded video reference and writes its own line, so the handover is ONE file, depth.mp4.
`binding.txt` exists for surfaces that DO have a per-reference binding field (the Ark
console is Variant B in prompt.txt). Until 2026-09-06 `Outbox/README.txt` said "take
exactly two files … paste binding.txt" and README.md's workflow step said the same, while
CLAUDE.md and the kit README said the opposite. Four surfaces, one sentence, held here.
"""

from __future__ import annotations

import re
from pathlib import Path

from depthc import kit as kitlib
from depthc.checks import check

ROOT = Path(__file__).resolve().parent.parent

# The contradiction, as an INSTRUCTION: paste binding.txt's line, or take two files.
CONTRADICTIONS = [
    re.compile(r"paste (its|the) (one )?line (from|into) [^\n]*binding", re.I),
    re.compile(r"paste the line from `?binding\.txt`?", re.I),
    re.compile(r"exactly two files", re.I),
]
# The truth, in the words every surface must carry.
ONE_FILE = "That is the whole handover"
NOT_OSIDE = re.compile(r"binding\.txt[^.]{0,60}surfaces that HAVE a per-reference binding field", re.I)


def _kit_readme() -> str:
    report = check(duration=5.0, width=1280, height=536, fps=24.0, size_bytes=800_000,
                   still_has_real_faces=False)
    manifest = kitlib.build_manifest(
        kit_name="Handover_001", source_path=Path("/tmp/x.mp4"), source_sha256="a" * 64,
        source_duration=8.0, source_fps=24.0, source_size=(1920, 1080), trim=(1.0, 6.0),
        crop=None, depth_duration=5.0, depth_fps=24.0, depth_size=(1280, 536),
        depth_bytes=800_000, depth_sha256="b" * 64, binding_sha256="c" * 64,
        frame_count=120, still_index=0, still_has_real_faces=False,
        tags=["handover"], description="", settings={}, report=report, device="cpu",
        elapsed=1.0,
    )
    return kitlib.render_kit_readme(manifest, report)


def _index(tmp_path: Path) -> str:
    return kitlib.write_outbox_index(tmp_path / "Outbox").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Prose wraps at ~90 columns on every surface; a phrase may cross a line break."""
    return re.sub(r"\s+", " ", text)


def surfaces(tmp_path: Path) -> dict[str, str]:
    raw = {
        "CLAUDE.md": (ROOT / "CLAUDE.md").read_text(encoding="utf-8"),
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "Outbox/README.txt": (ROOT / "Outbox" / "README.txt").read_text(encoding="utf-8"),
        "kit README.txt": _kit_readme(),
        "Outbox/INDEX.txt": _index(tmp_path),
    }
    return {name: _flat(text) for name, text in raw.items()}


def test_no_surface_tells_the_operator_to_paste_binding_txt_into_oside(tmp_path):
    for name, text in surfaces(tmp_path).items():
        for pattern in CONTRADICTIONS:
            hit = pattern.search(text)
            assert hit is None, f"{name}: {hit.group(0)!r}"


def test_every_surface_says_depth_mp4_is_the_whole_handover(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert ONE_FILE in text, f"{name} does not say {ONE_FILE!r}"
        assert "depth.mp4" in text, name


def test_every_surface_scopes_binding_txt_to_surfaces_with_a_binding_field(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert NOT_OSIDE.search(text), f"{name} does not scope binding.txt"


# --------------------------------------------- the door that makes the rule conditional
#
# "Paste no binding phrase in OSIDE" was stated flat, and it is only true of a HAND UPLOAD:
# an uploaded video ref has no binding field. OSIDE's Import blockout kit door — being built
# to accept depth kits — does have one and fills it from binding.txt itself, hash-checked
# against kit.json. Every surface has to name which path it is talking about, and none may
# state the door as today's behaviour, because the OSIDE side is not merged.

IMPORT_DOOR = re.compile(r"Import blockout kit", re.I)
HAND_UPLOAD = re.compile(r"hand upload|by hand|uploaded video ref", re.I)
NOT_MERGED = re.compile(r"not merged|being built|not .{0,20}behaviour today", re.I)


def test_every_surface_names_both_paths(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert IMPORT_DOOR.search(text), f"{name} does not name the import door"
        assert HAND_UPLOAD.search(text), f"{name} does not scope the rule to a hand upload"


def test_no_surface_states_the_import_door_as_todays_default(tmp_path):
    """The OSIDE side is not merged. A surface claiming the door already does this would
    send the operator looking for a button that is not there."""
    for name, text in surfaces(tmp_path).items():
        assert NOT_MERGED.search(text), f"{name} states the door without saying it is unbuilt"


def test_the_conditional_patterns_can_fire():
    """The sentence every surface used to carry must trip both discriminating greps."""
    old = ("In OSIDE, paste NO binding phrase - an uploaded video ref has no binding field "
           "and the Composer writes its own line.")
    assert not IMPORT_DOOR.search(old)
    assert not NOT_MERGED.search(old)
    assert HAND_UPLOAD.search(old)  # this one does NOT discriminate; the two above do


def test_the_contradiction_patterns_can_fire():
    """A grep that never matches anything is a claim: the old sentences must trip it."""
    old_outbox = "binding.txt   paste its one line into that reference binding field"
    old_readme = "upload `depth.mp4` as the video ref (the motion), and paste the line from `binding.txt`."
    assert any(p.search(old_outbox) for p in CONTRADICTIONS)
    assert any(p.search(old_readme) for p in CONTRADICTIONS)
    assert any(p.search("From every kit you take exactly two files to OSIDE") for p in CONTRADICTIONS)
