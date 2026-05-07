import sys
from pathlib import Path


def add_project_root_to_sys_path():
    """Add the project root to ``sys.path`` for direct script execution."""

    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    return project_root_str
