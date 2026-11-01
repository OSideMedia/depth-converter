# Security

## Reporting a problem

Please report it privately through GitHub: the **Security** tab of this repository, then
**Report a vulnerability**. Do not open a public issue for it.

## What this tool does, for judging a report

- It runs locally. The web UI binds to `127.0.0.1` only, and has no option to share it publicly.
- It makes no network calls when it runs, and has no upload code or telemetry. Gradio's analytics are off.
- `scripts/setup.sh` is the one part that downloads anything: a pinned `uv` from astral.sh, Python and
  the locked dependencies (hash-checked by `uv.lock`), Video Depth Anything at a pinned commit from
  GitHub, and the checkpoint from Hugging Face, verified against its SHA-256.
- The checkpoint is loaded with `torch.load(..., weights_only=True)`.
- It decodes video you give it with ffmpeg, so a hostile media file is ffmpeg's attack surface. Keep your
  ffmpeg current. It does not install or pin one.

There are no credentials anywhere in the tool, because it never talks to a paid API.
