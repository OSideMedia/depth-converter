"""One handover truth across every surface that states it (audit DC-2).

The verified fact (live run 2026-07-25): OSIDE's Composer has NO binding field for an
uploaded video reference and writes its own line, so the handover is ONE file, depth.mp4.
`binding.txt` exists for surfaces that DO have a per-reference binding field (the Ark
console is Variant B in prompt.txt). Until 2026-09-06 `Outbox/README.txt` said "take
exactly two files … paste binding.txt" and README.md's workflow step said the same, while
CLAUDE.md and the kit README said the opposite. Every surface, one sentence, held here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from depthc import kit as kitlib
from depthc.checks import check

ROOT = Path(__file__).resolve().parent.parent
PRE_AMENDMENT = "3f31315~1"  # the commit before PLAN.md's attach step was amended

# The contradiction, as an INSTRUCTION: paste binding.txt's line, or take two files.
CONTRADICTIONS = [
    re.compile(r"paste (its|the) (one )?line (from|into) [^\n]*binding", re.I),
    re.compile(r"paste the line from `?binding\.txt`?", re.I),
    re.compile(r"exactly two files", re.I),
    # PLAN.md's own phrasings, which the two greps above could not see (found 2026-09-16).
    re.compile(r"paste (its|the|the kit's)\b[^\n.]{0,40}binding (block|phrase|line)", re.I),
    re.compile(r"Variant A\W+FOR OSIDE COMPOSER", re.I),
    re.compile(r"skip the create-frame step", re.I),
]
# PLAN.md is a record: it keeps a superseded sentence and follows it with a dated note, so a hit
# struck that way is history, not an instruction. Anything further off is still live, and so is a
# hit followed by a note about something else. The note has to carry the law as well, because the
# file is held to every test below like any other surface. No other surface keeps history.
AMENDED = re.compile(r"\[AMENDED \d{4}-\d{2}-\d{2}")
ABOUT_THE_HANDOVER = re.compile(r"binding|create-frame|depth\.mp4|handover", re.I)
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
STRUCK_WITHIN = 400  # characters of flattened prose: the length of the longest such sentence


def _struck(text: str, end: int) -> bool:
    """A note strikes a hit when its own first sentence names the handover's subject."""
    for note in AMENDED.finditer(text, end, end + STRUCK_WITHIN):
        stop = SENTENCE_END.search(text, note.end())
        if ABOUT_THE_HANDOVER.search(text, note.end(), stop.start() if stop else len(text)):
            return True
    return False


def live_contradictions(text: str, *, record: bool = False) -> list[str]:
    hits = (m for pattern in CONTRADICTIONS for m in pattern.finditer(text))
    return [m.group(0) for m in hits if not (record and _struck(text, m.end()))]


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
        "PLAN.md": (ROOT / "PLAN.md").read_text(encoding="utf-8"),
        "Outbox/README.txt": (ROOT / "Outbox" / "README.txt").read_text(encoding="utf-8"),
        "kit README.txt": _kit_readme(),
        "Outbox/INDEX.txt": _index(tmp_path),
    }
    return {name: _flat(text) for name, text in raw.items()}


def test_no_surface_tells_the_operator_to_paste_binding_txt_into_oside(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert live_contradictions(text, record=name == "PLAN.md") == [], name


def test_every_surface_says_depth_mp4_is_the_whole_handover(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert ONE_FILE in text, f"{name} does not say {ONE_FILE!r}"
        assert "depth.mp4" in text, name


def test_every_surface_scopes_binding_txt_to_surfaces_with_a_binding_field(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert NOT_OSIDE.search(text), f"{name} does not scope binding.txt"


def test_the_kit_readme_gives_a_path_that_does_not_need_oside():
    text = _flat(_kit_readme())
    assert "TAKE THIS ONE FILE TO SEEDANCE" in text
    assert "Paste Variant B from prompt.txt" in text
    assert text.index("Variant B") < text.index("IN OSIDE")


# --------------------------------------------- the door that makes the rule conditional
#
# "Paste no binding phrase in OSIDE" was stated flat, and it is only true of a HAND UPLOAD:
# an uploaded video ref has no binding field. OSIDE's import door does have one and fills
# it from binding.txt itself, hash-checked against kit.json. Every surface has to name
# which path it is talking about.
#
# [AMENDED 2026-09-15] The door SHIPPED — OSIDE PR #278 (lib/import/depthKit.ts) — and the
# tile and dialog were renamed from "Import blockout kit" to "Import kit", because one door
# now takes both blockout kits and depth kits. Both of the things these surfaces used to say
# about it are now false and user-facing: the name, and "not merged yet". So the gate
# inverted. It used to REQUIRE the unbuilt caveat; it now REFUSES it, and refuses the old
# name outright — a surface still printing "Import blockout kit" sends the operator hunting
# a tile that no longer says that.

IMPORT_DOOR = re.compile(r"Import kit", re.I)
OLD_DOOR_NAME = re.compile(r"Import blockout kit", re.I)
HAND_UPLOAD = re.compile(r"hand upload|by hand|uploaded video ref", re.I)
# The caveats that were true until the door merged, and are now the stale claim to catch.
UNBUILT = re.compile(r"not merged|being built|not yet merged|until it ships|NOT merged as of", re.I)
# The door is merged: every surface has to say so, not merely stop saying it is not.
MERGED = re.compile(r"merged 2026-09-15|MERGED 2026-09-15|since 2026-09-15|takes depth kits since", re.I)


def test_every_surface_names_both_paths(tmp_path):
    for name, text in surfaces(tmp_path).items():
        assert IMPORT_DOOR.search(text), f"{name} does not name the import door"
        assert HAND_UPLOAD.search(text), f"{name} does not scope the rule to a hand upload"


def test_no_surface_still_calls_the_door_by_its_old_name(tmp_path):
    """The tile reads "Import kit". A surface saying "Import blockout kit" sends the
    operator looking for a label that is not on the screen any more."""
    for name, text in surfaces(tmp_path).items():
        hit = OLD_DOOR_NAME.search(text)
        assert hit is None, f"{name} still says {hit.group(0)!r}"


def test_no_surface_still_says_the_door_is_unbuilt(tmp_path):
    """The OSIDE side merged 2026-09-15. A surface still calling it unbuilt tells the
    operator to hand-upload when the door is right there — the inverse of the old defect."""
    for name, text in surfaces(tmp_path).items():
        hit = UNBUILT.search(text)
        assert hit is None, f"{name} still calls the door unbuilt: {hit.group(0)!r}"


def test_every_surface_says_the_door_is_merged(tmp_path):
    """Dropping the caveat is not the same as stating the fact; the date is what a reader
    can check against the OSIDE changelog."""
    for name, text in surfaces(tmp_path).items():
        assert MERGED.search(text), f"{name} names the door without saying it shipped"


def test_the_conditional_patterns_can_fire():
    """The sentence every surface carried BEFORE the door existed must trip the greps that
    discriminate, and must not trip the ones that do not."""
    old = ("In OSIDE, paste NO binding phrase - an uploaded video ref has no binding field "
           "and the Composer writes its own line.")
    assert not IMPORT_DOOR.search(old)
    assert not MERGED.search(old)
    assert HAND_UPLOAD.search(old)  # this one does NOT discriminate; the two above do


def test_the_stale_door_patterns_can_fire():
    """A grep that never matches anything is a claim. The exact sentences these surfaces
    carried until 2026-09-15 are kept here as the counterexample, so both refusals are
    proven to see the thing they refuse."""
    stale_readme = ("Through OSIDE's Import blockout kit door, which is being built to take "
                    "depth kits, the field is there and the door fills it from binding.txt.")
    stale_index = ("coming in through OSIDE's 'Import blockout kit' door, that door fills the "
                   "field from binding.txt itself, hash-checked against kit.json - the door's "
                   "behaviour, not merged yet.")
    stale_claude = ("Through OSIDE's Import blockout kit door - being built to take depth kits, "
                    "NOT merged as of 2026-09-15 - the field exists.")
    for stale in (stale_readme, stale_index, stale_claude):
        assert OLD_DOOR_NAME.search(stale), stale
        assert UNBUILT.search(stale), stale
        assert not MERGED.search(stale), stale
    # and the wording that replaced them trips neither refusal
    fixed = ("Coming in through OSIDE's 'Import kit' door - merged 2026-09-15 - you point it "
             "at the kit folder and that door fills the field from binding.txt itself.")
    assert IMPORT_DOOR.search(fixed)
    assert not OLD_DOOR_NAME.search(fixed)
    assert not UNBUILT.search(fixed)
    assert MERGED.search(fixed)


# ----------------------------------------------- the old name, everywhere it can be read
#
# surfaces() holds the five places that state the HANDOVER. The door's name is read in more
# places than that — the prompt templates the kit's prompt.txt is rendered from, the kit
# schema's own description, and the code comments that explain why binding.sha256 exists.
# The subject set is globbed, not listed, so a new template or module is in it the day it
# lands; CHANGELOG.md is out because its job is to record that the rename happened.
OLD_NAME_SUBJECTS = sorted(
    {
        *ROOT.glob("depthc/**/*.py"),
        *ROOT.glob("depthc/**/*.txt"),
        *ROOT.glob("depthc/**/*.json"),
        ROOT / "README.md",
        ROOT / "CLAUDE.md",
        ROOT / "Outbox" / "README.txt",
    }
)


def test_the_old_name_subject_set_is_not_empty():
    """A glob that matches nothing would make the sweep below pass by having nothing to
    read. Name the floor, so the gate fails if the tree moves under it."""
    names = {p.name for p in OLD_NAME_SUBJECTS}
    assert {"kit.py", "cli.py", "kit_schema.json", "seedance-2.5.txt", "seedance-2.0.txt"} <= names
    assert len(OLD_NAME_SUBJECTS) >= 10


def test_nothing_shipped_still_says_import_blockout_kit():
    """The rename is not a synonym: the tile reads "Import kit" and nothing else does."""
    for path in OLD_NAME_SUBJECTS:
        hit = OLD_DOOR_NAME.search(_flat(path.read_text(encoding="utf-8")))
        assert hit is None, f"{path.relative_to(ROOT)} still says {hit.group(0)!r}"


def test_the_contradiction_patterns_can_fire():
    """A grep that never matches anything is a claim: the old sentences must trip it."""
    old_outbox = "binding.txt   paste its one line into that reference binding field"
    old_readme = "upload `depth.mp4` as the video ref (the motion), and paste the line from `binding.txt`."
    assert any(p.search(old_outbox) for p in CONTRADICTIONS)
    assert any(p.search(old_readme) for p in CONTRADICTIONS)
    assert any(p.search("From every kit you take exactly two files to OSIDE") for p in CONTRADICTIONS)


def test_plans_own_phrasings_trip_the_gate_unless_a_dated_note_strikes_them():
    """The file as it stood before the 2026-09-16 amendment, sentence for sentence."""
    old_plan = [
        "upload `depth.mp4` as the video ref (the motion), paste the binding block from `prompt.txt`.",
        "**go straight to \"make video\" — SKIP the create-frame step.** A depth-motion shot",
        "**Variant A — FOR OSIDE COMPOSER (binding phrase only).** A bare noun phrase",
        "paste the kit's **Variant A binding phrase** into the ref's binding field",
    ]
    note = " > **[AMENDED 2026-09-16] Superseded: paste no binding phrase in OSIDE.**"
    for sentence in old_plan:
        assert live_contradictions(sentence, record=True), sentence
        assert live_contradictions(sentence + note, record=True) == [], sentence
        # too far away to be striking this sentence, and an undated note strikes nothing
        assert live_contradictions(sentence + " x" * STRUCK_WITHIN + note, record=True), sentence
        assert live_contradictions(sentence + " > **[AMENDED] Paste no binding phrase.**", record=True), sentence


def test_an_amendment_about_something_else_strikes_nothing():
    stale = "Take exactly two files to OSIDE."
    assert live_contradictions(stale + " [AMENDED 2026-11-01] Corrected a spelling elsewhere.", record=True)
    # Nor does the next paragraph's subject count as the note's.
    spill = stale + " [AMENDED 2026-11-01] Corrected a spelling elsewhere. Use binding.txt only in Ark."
    assert live_contradictions(spill, record=True)


def test_a_note_naming_the_one_file_handover_strikes_the_sentence():
    fixed = "Take exactly two files to OSIDE. [AMENDED 2026-11-01] The handover is ONE file: depth.mp4."
    assert live_contradictions(fixed, record=True) == []


def test_only_the_record_keeps_a_struck_sentence():
    # A note on a README sentence is not history. That file has to say the current thing.
    assert live_contradictions("Take exactly two files to OSIDE. [AMENDED 2026-11-01] No binding phrase.")


def test_plan_md_before_its_amendment_is_what_the_gate_was_missing():
    """The real file, not a paraphrase of it. Needs the commit in history, which CI's shallow
    checkout does not have, so it says so rather than passing by not looking."""
    shown = subprocess.run(["git", "show", f"{PRE_AMENDMENT}:PLAN.md"], cwd=ROOT, capture_output=True, text=True)
    if shown.returncode != 0:
        pytest.skip(f"{PRE_AMENDMENT} is not in this clone's history (a shallow checkout)")
    hits = live_contradictions(_flat(shown.stdout), record=True)
    assert any("binding block" in hit for hit in hits), hits
    assert any("create-frame" in hit for hit in hits), hits
