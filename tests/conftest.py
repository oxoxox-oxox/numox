from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_PKG = _SRC / "numox"


def _has_local_core_extension() -> bool:
    if not _PKG.is_dir():
        return False
    return any(_PKG.glob("_core*.pyd")) or any(_PKG.glob("_core*.so"))


if _has_local_core_extension():
    src_str = str(_SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
