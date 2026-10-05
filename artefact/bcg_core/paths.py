"""Repository paths, so relative config values mean the same thing wherever the app is started."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path) -> Path:
    """Return ``path`` unchanged if it is absolute, otherwise relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
