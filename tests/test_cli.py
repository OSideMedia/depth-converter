"""CLI-level regression pins.

Everything here is a validation path that dies BEFORE decode or inference, so the suite
stays fast — these exist because cli.py carried multiple fixed-regression comments with
no test holding any of them in place (2026-07-30 council audit).
"""

from __future__ import annotations

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
    """The 15s law fires on the trim AS GIVEN — snap must not clamp it legal first."""

    def test_long_trim_dies_before_snap(self, clip_long):
        result = runner.invoke(cli.app, ["kit", str(clip_long), "--trim", "0", "20"])
        assert result.exit_code == 1
        assert "caps video references at 15s" in _text(result)


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
