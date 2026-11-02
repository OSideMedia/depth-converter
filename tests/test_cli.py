"""CLI-level regression pins.

Everything here is a validation path that dies BEFORE decode or inference, so the suite
stays fast — these exist because cli.py carried multiple fixed-regression comments with
no test holding any of them in place (2026-07-30 council audit).
"""

from __future__ import annotations

import weakref
from pathlib import Path

import pytest
from typer.testing import CliRunner

from depthc import cli, memory
from depthc import motion as motionlib
from depthc.video_io import VideoIOError

runner = CliRunner()


def _text(result) -> str:
    """stdout plus stderr regardless of how this click version splits them."""
    out = result.output
    try:
        out += result.stderr
    except (ValueError, AttributeError):
        pass  # stderr not captured separately on this click version
    return out


class TestTrimValidation:
    """An impossible trim is a typo, not a request — snap must never launder it."""

    def test_negative_span_rejected(self, clip_3s):
        result = runner.invoke(cli.app, ["kit", str(clip_3s), "--trim", "2", "1"])
        assert result.exit_code == 1
        assert "not a window" in _text(result)

    def test_zero_span_rejected(self, clip_3s):
        result = runner.invoke(cli.app, ["kit", str(clip_3s), "--trim", "1", "1"])
        assert result.exit_code == 1
        assert "not a window" in _text(result)

    def test_negative_start_rejected(self, clip_3s):
        result = runner.invoke(cli.app, ["kit", str(clip_3s), "--trim", "-1", "2"])
        assert result.exit_code == 1
        assert "not a window" in _text(result)

    def test_beyond_eof_rejected(self, clip_3s):
        result = runner.invoke(cli.app, ["kit", str(clip_3s), "--trim", "1", "25"])
        assert result.exit_code == 1
        assert "past the end" in _text(result)

    def test_convert_validates_too(self, clip_3s):
        result = runner.invoke(cli.app, ["convert", str(clip_3s), "--trim", "2", "1"])
        assert result.exit_code == 1
        assert "not a window" in _text(result)


class TestConvertOverwriteGuard:
    """ffmpeg runs -y everywhere, so --out landing on the source would be silent and fatal."""

    def test_out_equal_to_input_refused(self, clip_3s):
        result = runner.invoke(
            cli.app, ["convert", str(clip_3s), "--out", str(clip_3s)]
        )
        assert result.exit_code == 1
        assert "overwrite the source" in _text(result)

    def test_relative_path_to_same_file_refused(self, clip_3s, monkeypatch):
        monkeypatch.chdir(clip_3s.parent)
        result = runner.invoke(cli.app, ["convert", str(clip_3s), "--out", clip_3s.name])
        assert result.exit_code == 1
        assert "overwrite the source" in _text(result)

    def test_audio_tmp_collision_refused(self, clip_3s, tmp_path):
        # convert result.mp4 --keep-audio writes result_audio.mp4 as a temp; a source by
        # that exact name must not be clobbered by its own conversion's scratch file.
        source = tmp_path / "clip_audio.mp4"
        source.write_bytes(clip_3s.read_bytes())
        result = runner.invoke(
            cli.app,
            ["convert", str(source), "--out", str(tmp_path / "clip.mp4"), "--keep-audio"],
        )
        assert result.exit_code == 1
        assert "overwrite the source" in _text(result)


class TestFifteenSecondGate:
    """The 15s law fires on the trim AS GIVEN — snap must not clamp it legal first.

    2.0 law, so 2.0 is passed explicitly: the default is 2.5 since 2026-09-06 (DC-5).
    """

    def test_long_trim_dies_before_snap(self, clip_long):
        result = runner.invoke(
            cli.app, ["kit", str(clip_long), "--trim", "0", "20", "--dialect", "seedance-2.0"]
        )
        assert result.exit_code == 1
        assert "caps video references at 15s" in _text(result)
        assert "rebuild with --dialect seedance-2.5" in _text(result)


class TestInboxHint:
    """The scout's copy-paste hint is what actually gets run — it must name the dialect.

    Until 2026-09-06 it printed no --dialect, so following the tool's own hint built a kit
    graded against whatever the default happened to be (audit DC-5).
    """

    def test_scout_hint_names_the_default_dialect_explicitly(self, capsys):
        from depthc import checks, motion as m
        import numpy as np

        profile = m.MotionProfile(fps=24.0, deltas=np.full(24 * 8, 3.0, dtype=np.float32))
        windows = m.score_windows(profile, 5.0)
        result = m.ScoutResult(
            profile=profile, windows=windows, per_second=list(profile.buckets),
            scout_fps=12.0, elapsed=1.0, target_seconds=5.0,
        )
        clip = Path("Inbox/x.mp4")
        cli._print_scout(clip, result)
        out = capsys.readouterr().out
        hint = next(line for line in out.splitlines() if "depthc kit" in line)
        assert f"--dialect {checks.DEFAULT_DIALECT}" in hint
        assert "--dialect seedance-2.5" in hint
        assert str(clip) in hint


class TestDialectPreflight:
    """A broken template must die before inference, naming the file, not after it."""

    def test_brace_template_dies_preflight(self, clip_3s, monkeypatch, tmp_path):
        from depthc import kit as kitlib

        bad = tmp_path / "templates"
        bad.mkdir()
        (bad / "broken.txt").write_text(
            "[[BINDING_A]]\nbind to {kit_name}\n[[/BINDING_A]]\n"
            'then literal JSON: {"motion": "copy"}\n'
        )
        monkeypatch.setattr(kitlib, "TEMPLATE_DIR", bad, raising=False)
        result = runner.invoke(
            cli.app, ["kit", str(clip_3s), "--dialect", "broken"]
        )
        assert result.exit_code == 1
        assert "not renderable" in _text(result)


def _fake_kit(
    tmp_path: Path,
    with_hash: bool = True,
    binding: str | None = "bind to the depth motion blueprint\n",
    with_binding_hash: bool = True,
) -> Path:
    """A kit folder with a valid kit.json and a depth.mp4 that is not a video, so any
    command reaching the decode dies there — everything BEFORE the decode is testable.

    `binding` is what lands in binding.txt (None writes no file at all); by default kit.json
    records the hash OF THAT TEXT, so the kit reads as intact and a test that wants a
    mismatch asks for one explicitly.
    """
    import hashlib
    import json

    from test_kit import _manifest

    from depthc import kit as kitlib

    kit_dir = tmp_path / "Old_001"
    kit_dir.mkdir()
    if binding is not None:
        (kit_dir / kitlib.BINDING_FILE).write_text(binding)
    manifest = _manifest(
        binding_sha256=hashlib.sha256((binding or "").encode()).hexdigest()
    )
    if not with_hash:
        del manifest["depth"]["sha256"]
    if not with_binding_hash:
        del manifest["binding"]["sha256"]
    (kit_dir / kitlib.KIT_MANIFEST).write_text(json.dumps(manifest))
    (kit_dir / "depth.mp4").write_bytes(b"not a video")
    return kit_dir


class TestCompareNamesAMissingDepthHash:
    """The content join needs kit.json.depth.sha256; a kit built before 2026-09-06 has
    none. compare says so ONCE, before the depth pass, and carries on — it never refuses
    a real kit with real fires behind it."""

    def _kit(self, tmp_path: Path, with_hash: bool) -> Path:
        return _fake_kit(tmp_path, with_hash=with_hash)

    def test_old_kit_is_named_once_then_compare_proceeds(self, tmp_path):
        kit_dir = self._kit(tmp_path, with_hash=False)
        result = runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")])
        text = _text(result)
        assert text.count("no depth.sha256") == 1
        assert "refs[].sha256" in text
        assert result.exit_code == 1  # the fake depth.mp4, not the missing hash

    def test_a_hashed_kit_gets_no_note(self, tmp_path):
        kit_dir = self._kit(tmp_path, with_hash=True)
        result = runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")])
        assert "no depth.sha256" not in _text(result)


class TestEverySurfaceVerifiesTheBinding:
    """A kit's WORDS have to belong to its clip. OSIDE's Import kit door fills a
    reference's binding field from binding.txt, so every command that reads a finished kit
    checks that file against the hash kit.json recorded — the blockout kit's clip+legend
    pair is what caught one kit's words riding on another kit's clip.

    The FAIL never destroys evidence: `compare` prints its measurement first and `result`
    writes the fire first; the exit code is what carries the failure out.
    """

    def _tampered(self, tmp_path: Path) -> Path:
        kit_dir = _fake_kit(tmp_path)
        (kit_dir / "binding.txt").write_text("someone else's words\n")
        return kit_dir

    def test_compare_names_the_mismatch_and_exits_nonzero(self, tmp_path):
        kit_dir = self._tampered(tmp_path)
        result = runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")])
        text = _text(result)
        assert "[FAIL] binding" in text
        assert "does not match" in text
        assert "templates/" in text and "rebuild" in text
        # The exit code here is the fake depth.mp4's, not the binding's — what discriminates
        # is the text, against `test_compare_is_silent_on_an_intact_kit` below. The binding's
        # own exit path is proven end to end in test_compare_measures_first_then_fails.
        assert result.exit_code == 1

    def test_compare_is_silent_on_an_intact_kit(self, tmp_path):
        kit_dir = _fake_kit(tmp_path)
        text = _text(runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")]))
        assert "binding" not in text

    def test_compare_names_a_kit_built_before_the_key_existed(self, tmp_path):
        """Reported the way a missing depth.sha256 is — never invented, never a FAIL."""
        kit_dir = _fake_kit(tmp_path, with_binding_hash=False)
        text = _text(runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")]))
        assert text.count("no binding.sha256") == 1
        assert "2026-09-15" in text and "rebuild the kit" in text
        assert "[FAIL] binding" not in text

    def test_result_records_the_fire_first_then_fails(self, tmp_path):
        """A generation is paid evidence: a broken text file must never cost a fire record."""
        import json

        kit_dir = self._tampered(tmp_path)
        result = runner.invoke(cli.app, ["result", str(kit_dir), "--verdict", "adhered"])
        saved = json.loads((kit_dir / "result.json").read_text())
        assert saved["fires"][0]["verdict"] == "adhered"  # evidence landed
        assert "[FAIL] binding" in _text(result)
        assert result.exit_code == 1  # and the command still says the kit is broken

    def test_result_on_an_intact_kit_succeeds(self, tmp_path):
        kit_dir = _fake_kit(tmp_path)
        result = runner.invoke(cli.app, ["result", str(kit_dir), "--verdict", "adhered"])
        assert result.exit_code == 0, _text(result)
        assert "binding" not in _text(result)

    @pytest.mark.slow
    def test_compare_measures_first_then_fails(self, clip_3s, tmp_path):
        """A real kit, a real compare: the measurement is on screen AND the exit is 1. A
        FAIL that swallowed the numbers would cost the operator the fire they paid for."""
        lib = tmp_path / "lib"
        built = runner.invoke(
            cli.app, ["kit", str(clip_3s), "--library", str(lib), "--tags", "bindfail"]
        )
        kit_dir = next((p for p in lib.iterdir() if p.is_dir()), None)
        assert kit_dir is not None, _text(built)

        clean = runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")])
        assert clean.exit_code == 0, _text(clean)
        assert "record it:" in _text(clean)

        (kit_dir / "binding.txt").write_text("someone else's words\n")
        tampered = runner.invoke(cli.app, ["compare", str(kit_dir), str(kit_dir / "depth.mp4")])
        assert "[FAIL] binding" in _text(tampered)
        assert "record it:" in _text(tampered)  # the measurement survived the failure
        assert tampered.exit_code == 1

    def test_a_missing_binding_file_is_the_same_failure(self, tmp_path):
        kit_dir = _fake_kit(tmp_path)
        (kit_dir / "binding.txt").unlink()
        result = runner.invoke(cli.app, ["result", str(kit_dir), "--verdict", "adhered"])
        assert "[FAIL] binding" in _text(result) and "not here" in _text(result)
        assert result.exit_code == 1


def test_compare_lets_go_of_the_reference_before_the_generated_clips_pass(clip_tiny, tmp_path, monkeypatch):
    """The memory check credits what this process holds as room the pass will reuse, so a reference
    still alive during the pass would be counted as room it is not."""
    import shutil

    kit_dir = _fake_kit(tmp_path)
    shutil.copy(clip_tiny, kit_dir / "depth.mp4")
    seen = {}
    real_measure = motionlib.measure

    def measure(frames, fps):
        seen["reference"] = weakref.ref(frames)
        return real_measure(frames, fps)

    def depth_pass(*_a, **_k):
        seen["alive during the pass"] = seen["reference"]() is not None
        raise VideoIOError("stop here")

    monkeypatch.setattr(motionlib, "measure", measure)
    monkeypatch.setattr(cli, "_depth_profile", depth_pass)
    result = runner.invoke(cli.app, ["compare", str(kit_dir), str(clip_tiny)])
    assert seen.get("alive during the pass") is False, _text(result)


def test_a_clip_too_big_to_read_reads_as_plain_words(clip_tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "_SMALL_READ", 0)
    monkeypatch.setenv(memory.OVERRIDE_ENV, "0.000001")
    result = runner.invoke(cli.app, ["convert", str(clip_tiny), "--out", str(tmp_path / "out.mp4")])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), repr(result.exception)
    assert "to read this clip" in _text(result)


class TestGenerationFlag:
    """`--generation <id>` on `result` and `compare` names the OSIDE generation a fire
    became (PLAN-SHARED-GENERATION-ID step d). `compare` never writes the fire — the
    verdict is human-only — so its id is validated up front and rides into the
    ready-to-paste `depthc result` line, which is what writes fires[].generationId."""

    GEN = "3f2504e0-4f89-4d3c-9a6b-0305e82c3301"

    def test_result_writes_the_generation_into_the_fire_entry(self, tmp_path):
        import json

        kit_dir = _fake_kit(tmp_path)
        result = runner.invoke(
            cli.app, ["result", str(kit_dir), "--verdict", "adhered", "--generation", self.GEN]
        )
        assert result.exit_code == 0, _text(result)
        saved = json.loads((kit_dir / "result.json").read_text())
        assert saved["fires"][0]["generationId"] == self.GEN

    def test_result_refuses_a_non_uuid_naming_the_grammar(self, tmp_path):
        kit_dir = _fake_kit(tmp_path)
        result = runner.invoke(
            cli.app, ["result", str(kit_dir), "--verdict", "adhered", "--generation", "gen-42"]
        )
        assert result.exit_code == 1
        text = _text(result)
        assert "UUID v4" in text and "8-4-4-4-12" in text
        assert not (kit_dir / "result.json").exists()

    def test_compare_refuses_a_non_uuid_before_the_depth_pass(self, tmp_path):
        kit_dir = _fake_kit(tmp_path)
        args = ["compare", str(kit_dir), str(kit_dir / "depth.mp4"), "--generation"]
        bad = runner.invoke(cli.app, args + ["gen-42"])
        assert bad.exit_code == 1 and "UUID v4" in _text(bad)
        # A well-formed id gets past the grammar and dies on the fake decode instead —
        # that is what proves the grammar check ran FIRST.
        good = runner.invoke(cli.app, args + [self.GEN])
        assert good.exit_code == 1 and "UUID v4" not in _text(good)

    def test_compare_paste_line_carries_the_generation(self):
        from types import SimpleNamespace

        c = SimpleNamespace(suggestion="adhered", overlap_seconds=5.0, ratio=0.93, r0=0.82)
        hint = cli._record_hint("Kit_001", c, self.GEN)
        assert "depthc result" in hint and f"--generation {self.GEN}" in hint
        assert "--generation" not in cli._record_hint("Kit_001", c, None)


class TestNotEnoughMemory:
    """Every command that runs a depth pass gives the same plain refusal, before the model
    loads. A traceback here is what a person on a smaller machine would otherwise read."""

    @pytest.fixture(autouse=True)
    def _machine_with_nothing_to_spare(self, monkeypatch):
        monkeypatch.setenv(memory.OVERRIDE_ENV, "1")  # enough to read these small clips, not to run

    @staticmethod
    def _refused(result) -> None:
        text = _text(result)
        assert result.exit_code == 1, text
        assert isinstance(result.exception, SystemExit), repr(result.exception)
        assert "Not enough free memory for this depth pass" in text and "Close other apps" in text
        assert "Traceback" not in text

    def test_convert(self, clip_tiny, tmp_path):
        self._refused(runner.invoke(cli.app, ["convert", str(clip_tiny), "--out", str(tmp_path / "d.mp4")]))
        assert not (tmp_path / "d.mp4").exists()

    def test_kit_leaves_no_half_built_folder(self, clip_tiny, tmp_path):
        lib = tmp_path / "lib"
        self._refused(runner.invoke(cli.app, ["kit", str(clip_tiny), "--library", str(lib), "--tags", "x"]))
        assert not lib.exists() or not [p for p in lib.iterdir() if p.is_dir()]

    def test_describe(self, clip_tiny):
        self._refused(runner.invoke(cli.app, ["describe", str(clip_tiny)]))

    def test_scout(self, clip_tiny):
        self._refused(runner.invoke(cli.app, ["scout", str(clip_tiny)]))

    def test_compare(self, clip_tiny, tmp_path):
        import shutil

        kit_dir = _fake_kit(tmp_path)
        shutil.copy(clip_tiny, kit_dir / "depth.mp4")
        self._refused(runner.invoke(cli.app, ["compare", str(kit_dir), str(clip_tiny)]))

    def test_inbox_says_so_for_that_clip_and_carries_on(self, clip_3s, tmp_path):
        """The triage list is still worth reading when its scout pass cannot run."""
        import shutil

        shutil.copy(clip_3s, tmp_path / "one.mp4")  # under 2 s never reaches the scout
        result = runner.invoke(cli.app, ["inbox", "--inbox", str(tmp_path)])
        text = _text(result)
        assert result.exit_code == 0, text
        assert "scout failed" in text and "Not enough free memory" in text
