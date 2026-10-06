import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from src.config_loader import load_config
from src.utils import setup_logging

from src.split_dataset import split_dataset


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Tomato classifier: split dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--allow-fallback-groups", action="store_true")
    p.add_argument("--allow-small-test-set", action="store_true")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    c = load_config(a.config)
    try:
        result = split_dataset(
            c,
            dry_run=a.dry_run,
            force_rebuild=a.force_rebuild,
            allow_fallback_groups=a.allow_fallback_groups,
            allow_small_test_set=a.allow_small_test_set,
            seed=a.seed,
        )
    except (ValueError, RuntimeError, OSError, PermissionError) as e:
        p.exit(2, "ERROR: " + str(e) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
