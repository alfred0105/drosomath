"""DrosoMath FastAPI entrypoint.

The active experiment server lives in ``stage3s1_server`` so future stage swaps can
keep the public uvicorn command stable: ``uvicorn app.main:app``.
"""

from .stage3s1_server import app

__all__ = ["app"]
