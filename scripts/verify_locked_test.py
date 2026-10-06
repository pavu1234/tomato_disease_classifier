import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from src.config_loader import load_config
from src.utils import setup_logging

from src.split_dataset import verify_lock


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Tomato classifier: verify locked test")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    c = load_config(a.config)
    try:
        result = verify_lock(c)
    except (ValueError, RuntimeError, OSError, PermissionError) as e:
        p.exit(2, "ERROR: " + str(e) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
