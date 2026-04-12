"""
shared.py — EpochDB example/benchmark shared utilities
"""
import os


def load_dotenv(start_path: str = None) -> None:
    """
    Load variables from a .env file into os.environ without requiring
    the python-dotenv package.

    Walks upward from `start_path` (defaults to the calling file's directory)
    looking for the first .env file found.
    """
    search = os.path.abspath(start_path or os.getcwd())
    for _ in range(5):  # Walk up at most 5 levels.
        candidate = os.path.join(search, ".env")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
            return
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
