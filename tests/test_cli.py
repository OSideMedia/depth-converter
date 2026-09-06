"""CLI-level regression pins.

Everything here is a validation path that dies BEFORE decode or inference, so the suite
stays fast — these exist because cli.py carried multiple fixed-regression comments with
no test holding any of them in place (2026-07-30 council audit).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from depthc import cli

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


def _fake_kit(tmp_path: Path, with_hash: bool = True) -> Path:
    """A kit folder with a valid kit.json and a depth.mp4 that is not a video, so any
    command reaching the decode dies there — everything BEFORE the decode is testable."""
    import json

    from test_kit import _manifest

    from depthc import kit as kitlib

    kit_dir = tmp_path / "Old_001"
    kit_dir.mkdir()
    manifest = _manifest()
    if not with_hash:
        del manifest["depth"]["sha256"]
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
