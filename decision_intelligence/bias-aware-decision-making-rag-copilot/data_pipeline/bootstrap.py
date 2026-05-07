"""
Lightweight bootstrap so pipeline scripts run from any context.

The pipeline scripts are designed to work whether they are invoked as
``python data_pipeline/foo.py`` (PyCharm "Run File", terminal), as
``python -m data_pipeline.foo``, or imported as a module. The first form does
not put the project root on ``sys.path``; this helper patches it in.
"""

import sys
from pathlib import Path


def add_project_root_to_sys_path():
    """Ensure the project root is on ``sys.path`` and return its string path.

    Idempotent: if the root is already present (module/package execution path)
    the function is effectively a no-op. Returns the project root so callers
    that need it for path resolution do not have to recompute it.
    """

    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root_str
