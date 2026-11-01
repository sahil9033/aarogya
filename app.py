"""Process entry point for hosts that run ``python app.py``.

Render's default start command for a Python web service is ``python app.py``,
so a service created without an explicit start command will run this file. It
contains no application logic -- it hands the process to uvicorn, and the ASGI
app itself lives in ``backend/app``.

If you would rather not keep this shim, set the service's Start Command to:

    cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"


def main() -> None:
    os.chdir(BACKEND)

    # ``backend`` has to precede the repository root on the import path, or
    # ``import app`` would find this file instead of the package beside it.
    sys.path = [str(BACKEND)] + [
        entry for entry in sys.path if entry and Path(entry).resolve() != ROOT
    ]

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
