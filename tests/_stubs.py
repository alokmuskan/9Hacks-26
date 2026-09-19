"""Shared test stubs for the heavy optional dependencies.

The backend pipeline imports `cv2` and `insightface` at module scope. Neither is
needed to exercise the logic under test, so the suite installs lightweight
stand-ins into `sys.modules` before importing `main` / `server`.

Every test module used to carry its own near-duplicate copy of these stubs.
Because `cv2` is cached in `sys.modules` on first insert, whichever copy ran
first won -- and a partial copy that happened to win broke every later test.
This module is the single source of truth, and the stub is a superset of
everything the pipeline actually calls.

Call `install()` in `setUp`, never at module import time: test discovery imports
every module before running any test, so a module-level install would claim
`sys.modules["cv2"]` first, outside any test.
"""

import sys
import types

import numpy as np

__all__ = ["MISSING", "install", "install_cv2", "install_insightface"]

#: Names the pipeline calls on `cv2`. Asserted complete by the stub guard test.
CV2_CONSTANTS = {
    "CAP_V4L2": 200,
    "CAP_ANY": 0,
    "CAP_FFMPEG": 1900,
    "CAP_PROP_BUFFERSIZE": 38,
    "CAP_PROP_FRAME_WIDTH": 3,
    "CAP_PROP_FRAME_HEIGHT": 4,
    "FONT_HERSHEY_SIMPLEX": 0,
    "LINE_AA": 16,
    "WINDOW_NORMAL": 0,
    "INTER_LINEAR": 1,
    "INTER_AREA": 3,
    "COLOR_BGR2RGB": 4,
    "IMWRITE_JPEG_QUALITY": 1,
}

#: Functions the pipeline calls on `cv2`. Keep this list and `_build_cv2` in step.
CV2_FUNCTIONS = (
    "cvtColor",
    "resize",
    "rectangle",
    "addWeighted",
    "putText",
    "polylines",
    "line",
    "circle",
    "getTextSize",
    "imwrite",
    "imencode",
    "imshow",
    "waitKey",
    "namedWindow",
    "destroyAllWindows",
    "setLogLevel",
    "VideoCapture",
)

MISSING = object()


def _build_cv2() -> types.ModuleType:
    stub = types.ModuleType("cv2")
    for name, value in CV2_CONSTANTS.items():
        setattr(stub, name, value)

    stub.cvtColor = lambda frame, _mode: frame
    stub.resize = lambda frame, *_args, **_kwargs: frame
    stub.rectangle = lambda *_args, **_kwargs: None
    stub.addWeighted = lambda *_args, **_kwargs: None
    stub.putText = lambda *_args, **_kwargs: None
    stub.polylines = lambda *_args, **_kwargs: None
    stub.line = lambda *_args, **_kwargs: None
    stub.circle = lambda *_args, **_kwargs: None
    stub.getTextSize = lambda text, *_args, **_kwargs: ((len(text) * 8, 12), 2)
    stub.imwrite = lambda *_args, **_kwargs: True
    stub.imencode = lambda *_args, **_kwargs: (True, np.zeros(4, dtype=np.uint8))
    stub.imshow = lambda *_args, **_kwargs: None
    stub.waitKey = lambda *_args, **_kwargs: -1
    stub.namedWindow = lambda *_args, **_kwargs: None
    stub.destroyAllWindows = lambda: None
    stub.setLogLevel = lambda *_args, **_kwargs: None
    stub.VideoCapture = lambda *_args, **_kwargs: None
    return stub


def _build_insightface() -> tuple[types.ModuleType, types.ModuleType]:
    insightface_stub = types.ModuleType("insightface")
    app_stub = types.ModuleType("insightface.app")

    class _FaceAnalysis:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return []

    app_stub.FaceAnalysis = _FaceAnalysis
    insightface_stub.app = app_stub
    return insightface_stub, app_stub


def install_cv2(force: bool = False) -> types.ModuleType:
    """Install the shared `cv2` stub unless a real/later module already claimed it."""
    if force or "cv2" not in sys.modules:
        sys.modules["cv2"] = _build_cv2()
    return sys.modules["cv2"]


def install_insightface(force: bool = False) -> None:
    if force or "insightface" not in sys.modules:
        insightface_stub, app_stub = _build_insightface()
        sys.modules["insightface"] = insightface_stub
        sys.modules["insightface.app"] = app_stub


def install(force: bool = False) -> None:
    """Install both stubs. Idempotent, so calling it from every `setUp` is cheap."""
    install_cv2(force=force)
    install_insightface(force=force)
