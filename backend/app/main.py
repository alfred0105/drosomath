"""DrosoMath FastAPI entrypoint.

The active experiment server lives in ``stage3s3_server`` while the public
uvicorn command stays stable: ``uvicorn app.main:app``.
"""

from .stage3s3_server import app

__all__ = ["app"]
