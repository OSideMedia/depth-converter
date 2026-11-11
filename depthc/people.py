"""Where the people are in a picture, from Apple Vision. Optional, and macOS only.

The camera reader (`motion.camera_reading`) drops the tracked points that sit on people before it
fits the camera, and the scout (`motion.scout_clip`) counts the people in frame. Both work without
this module's answer: off macOS, or on a Mac older than macOS 14, `find` returns None and the
reader behaves as it did before people were masked (with a depth fallback where it has a depth).

WHY VISION. The second real-footage score failed where a subject filled the frame: dancers in a
vertical frame against a plain wall, a person walking away close to the lens. Their points outvote
the wall's and the subject's motion is named as a camera move. A person mask removes them from the
vote. Vision's person instance mask is a framework that ships with macOS: no weights to download
and no model licence to carry, which is why it was allowed when a second model was not (the VDA-Small
law in CLAUDE.md). It is reached through PyObjC (MIT), a dependency installed on macOS only.

WHAT IT CANNOT DO. It finds people, and only people: a train, a car or a flower that fills the frame
is not masked. The model ships with the OS, so a macOS update can change what it finds. It returns at
most four people per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Vision is handed each frame at this long edge. Its instance mask comes back at its own size
# (512 on the long side on macOS 26), so a larger frame buys nothing and costs time.
MASK_LONG_EDGE = 512
# A person smaller than this share of the frame is not counted as a figure. A speck in a crowd is
# not a second subject competing with the first.
# ponytail: a fixed area floor, chosen by eye on the seen dance and street clips; a figure's motion
# share (not its size) is the upgrade if the count ever votes in anything.
MIN_PERSON_SHARE = 0.005
# An outline counts as a person when at least this share of it lies inside a box the body detector
# drew (see `_labels`). Half: a raised arm or a skirt can leave the box, a hallucinated outline
# with no box around it scores zero.
BOX_OVERLAP = 0.5


@dataclass
class People:
    """Per frame: a boolean person mask at the size asked for, and how many people Vision counted."""

    masks: np.ndarray   # [N, H, W] bool
    counts: np.ndarray  # [N] int, people at least MIN_PERSON_SHARE of the frame

    @property
    def share(self) -> float:
        """Share of the picture covered by people, over the whole clip."""
        return float(self.masks.mean()) if self.masks.size else 0.0


def available() -> bool:
    """True when Vision's person instance mask can run here (macOS 14 or later, PyObjC installed)."""
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        from Foundation import NSData  # noqa: F401
    except ImportError:
        return False
    return hasattr(Vision, "VNGeneratePersonInstanceMaskRequest")


def _labels(rgb: np.ndarray) -> np.ndarray | None:
    """One RGB frame through Vision: an instance label image (0 background, 1..n people), or None.

    Two requests on the same frame. The instance mask draws each person's outline, but it is made
    to cut people OUT of a picture that has them, and on a picture with none it outlines something
    anyway, with full confidence: on the seen clips it covered 99% of a flower and most of a forest.
    So an instance is kept only where the human-rectangle detector found a body: at least
    BOX_OVERLAP of its pixels must sit inside a detected box. That detector found nobody on the
    flower or the forest and found every dancer and walker checked by eye.
    """
    import cv2
    import objc
    import Quartz
    import Vision
    from Foundation import NSData

    ok, encoded = cv2.imencode(".bmp", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    height, width = rgb.shape[:2]
    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(encoded.tobytes(), len(encoded))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        outline = Vision.VNGeneratePersonInstanceMaskRequest.alloc().init()
        bodies = Vision.VNDetectHumanRectanglesRequest.alloc().init()
        bodies.setUpperBodyOnly_(False)
        done, _error = handler.performRequests_error_([outline, bodies], None)
        if not done:
            return None
        boxes = np.zeros((height, width), dtype=bool)
        for body in bodies.results() or []:
            box = body.boundingBox()  # normalised, origin at the bottom left
            x0 = int(np.floor(box.origin.x * width))
            x1 = int(np.ceil((box.origin.x + box.size.width) * width))
            y0 = int(np.floor((1.0 - box.origin.y - box.size.height) * height))
            y1 = int(np.ceil((1.0 - box.origin.y) * height))
            boxes[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        results = outline.results() or []
        if not boxes.any():
            return np.zeros((height, width), dtype=np.uint8)
        if not len(results):
            # A body with no outline: the box itself is the best mask there is.
            return boxes.astype(np.uint8)
        buffer = results[0].instanceMask()
        Quartz.CVPixelBufferLockBaseAddress(buffer, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            rows = Quartz.CVPixelBufferGetHeight(buffer)
            cols = Quartz.CVPixelBufferGetWidth(buffer)
            stride = Quartz.CVPixelBufferGetBytesPerRow(buffer)
            raw = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(stride * rows)
            labels = np.frombuffer(raw, dtype=np.uint8).reshape(rows, stride)[:, :cols].copy()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buffer, Quartz.kCVPixelBufferLock_ReadOnly)
    # The mask is in the frame's own normalised coordinates, whatever its pixel size.
    labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
    kept = np.zeros_like(labels)
    for label in range(1, int(labels.max()) + 1):
        pixels = labels == label
        if pixels.any() and float(boxes[pixels].mean()) >= BOX_OVERLAP:
            kept[pixels] = label
    return kept


def find(picture: np.ndarray, size: tuple[int, int] | None = None) -> People | None:
    """Person masks and counts for every frame of `picture` ([N,H,W,3] RGB or [N,H,W] grey).

    `size` is the (width, height) the masks come back at; the picture's own size when None. None
    when Vision cannot run here, or failed on any frame: the caller then reads without a person
    mask, which is the documented fallback, never an error.
    """
    import cv2

    if not available() or picture.ndim not in (3, 4) or not picture.shape[0]:
        return None
    height, width = picture.shape[1:3]
    out_w, out_h = size or (width, height)
    scale = min(1.0, MASK_LONG_EDGE / max(height, width))
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    masks = np.zeros((picture.shape[0], out_h, out_w), dtype=bool)
    counts = np.zeros(picture.shape[0], dtype=np.int32)
    for index, frame in enumerate(picture):
        rgb = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if scale < 1.0:
            rgb = cv2.resize(rgb, small, interpolation=cv2.INTER_AREA)
        try:
            labels = _labels(np.ascontiguousarray(rgb))
        except Exception:  # noqa: BLE001 — any Vision failure means "no mask", never a crash
            return None
        if labels is None:
            return None
        areas = np.bincount(labels.ravel(), minlength=2)[1:]
        counts[index] = int((areas >= MIN_PERSON_SHARE * labels.size).sum())
        masks[index] = cv2.resize(
            (labels > 0).astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return People(masks=masks, counts=counts)
