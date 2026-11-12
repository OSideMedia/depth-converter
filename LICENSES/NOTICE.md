# Third-party notices

## Video-Depth-Anything

Copyright (2025) Bytedance Ltd. and/or its affiliates.
Licensed under the Apache License, Version 2.0 — full text in
`Video-Depth-Anything-LICENSE.txt`.

- Source: https://github.com/DepthAnything/Video-Depth-Anything
- Pinned commit: `4f5ae23172ba60fd7bc11ef671cca678842c7072`
- Paper: *Video Depth Anything: Consistent Depth Estimation for Super-Long Videos*
  (CVPR 2025), arXiv:2501.12375

The upstream repository is vendored unmodified by `scripts/setup.sh`. Depth Converter
imports its model classes and supplies its own device handling, video I/O, and encoding.

### Checkpoint — VDA-Small only

Depth Converter uses `video_depth_anything_vits.pth` from the Hugging Face repo
`depth-anything/Video-Depth-Anything-Small`, which is released under **Apache-2.0**.

> **LICENSE LAW.** Across Depth Anything V2 and Video Depth Anything, **only the Small
> models are Apache-2.0. Every Base / Large / Giant checkpoint there is CC-BY-NC-4.0
> (non-commercial).** Depth Anything 3 is licensed model by model (DA3-SMALL and DA3-BASE
> Apache-2.0; DA3-LARGE, DA3-GIANT and the nested models CC-BY-NC-4.0, read 2026-11-11) and
> none of it ships here. This tool is used for commercial film work, so it ships
> the Small model exclusively and deliberately offers **no model-size selector**. Adding one
> that downloads Base/Large/Giant weights would be a licensing violation, not an upgrade.

### Images in the README

`DOCS/images/banner.png` and `DOCS/images/kit-contents.png` are illustrations made with an AI
image generator for this project. They explain the idea and are not output of the tool.
`DOCS/images/real-source-and-depth.jpg` and `DOCS/images/real-contact-sheet.jpg` are the tool's
own output from a blockout the project rendered itself in Blender (kit `Clay-close-alone_001`).
