from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None, root=None):
    root = Path(root or ROOT).resolve()
    with open(path or root / "config/config.yaml", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    if set(c["classes"]["expected"]) != {"early_blight", "healthy", "late_blight"}:
        raise ValueError("Exactly early_blight, healthy, late_blight are required")
    ratios = [c["split"][k + "_ratio"] for k in ["train", "val", "test"]]
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Split ratios must be positive and sum to 1")
    if c["model"]["architecture"] not in ("MobileNetV3Small", "MobileNetV2"):
        raise ValueError("Unsupported architecture")
    if c["model"]["pretrained_weights"] != "imagenet":
        raise ValueError("ImageNet initialization is mandatory")
    if (
        c["image"]["channels"] != 3
        or min(c["image"]["height"], c["image"]["width"]) < 32
    ):
        raise ValueError("RGB inputs with dimensions >=32 required")
    for k in ["batch_size", "stage1_epochs", "fine_tune_epochs"]:
        if c["training"][k] < 1:
            raise ValueError(k + " must be positive")
    if not 0 <= c["split"]["duplicate_hamming_threshold"] <= 64:
        raise ValueError("Invalid perceptual hash threshold")
    c["root"] = str(root)
    paths = {k: (root / v).resolve() for k, v in c["paths"].items()}
    if any(not p.is_relative_to(root) for p in paths.values()):
        raise ValueError("All configured paths must remain within the project")
    dirs = [
        paths[k] for k in ["raw_data_dir", "train_dir", "val_dir", "locked_test_dir"]
    ]
    if any(
        a == b or a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(dirs)
        for b in dirs[i + 1 :]
    ):
        raise ValueError("Raw and split directories must be disjoint")
    return c


def path_for(c, name):
    return Path(c["root"]) / c["paths"][name]
