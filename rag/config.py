import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# 5-line .env loader, swap for python-dotenv if quoting/expansion ever needed
if (ROOT / ".env").exists():
    for _line in (ROOT / ".env").read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

VAULT = Path(os.environ.get("VAULT", ROOT / "vault"))
INDEX_NPZ = ROOT / "index.npz"
INDEX_JSON = ROOT / "index.json"
MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5
MAX_CHUNK = 1500
