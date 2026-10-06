import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from src.config_loader import load_config
from src.utils import setup_logging

from src.train import train


def main():
    setup_logging()
    p = argparse.ArgumentParser(description="Tomato classifier: train model")
    p.add_argument("--config", default=None)
    p.add_argument("--stage", choices=["all", "stage1"], default="all")
    p.add_argument("--skip-fine-tune", action="store_true")
    p.add_argument("--stage1-epochs", type=int)
    p.add_argument("--fine-tune-epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    c = load_config(a.config)
    try:
        for key in ["stage1_epochs", "fine_tune_epochs", "batch_size"]:
            value = getattr(a, key)
            if value is not None:
                if value < 1:
                    raise ValueError(key + " must be positive")
                c["training"][key] = value
        result = train(
            c, stage=a.stage, skip_fine_tune=a.skip_fine_tune, resume=a.resume
        )
    except (ValueError, RuntimeError, OSError, PermissionError) as e:
        p.exit(2, "ERROR: " + str(e) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
