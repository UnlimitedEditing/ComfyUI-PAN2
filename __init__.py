import os
import subprocess

# Graydient keeps the clone made when a project directory was first created. Pull before importing
# nodes so a push to this repo takes effect without re-restoring the workflow (best effort).
try:
    _here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(_here, ".git")):
        subprocess.run(["git", "-C", _here, "pull", "--ff-only", "-q"], timeout=40,
                       capture_output=True, check=False)
except Exception:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
