import shutil
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from src.config_loader import load_config
from src.manifest import build_manifest
from src.split_dataset import split_dataset


@pytest.fixture
def dataset(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config").mkdir()
    shutil.copy2(root / "config/config.yaml", tmp_path / "config/config.yaml")
    c = load_config(root=tmp_path)
    rng = np.random.default_rng(137)
    for label in c["classes"]["expected"]:
        folder = tmp_path / "data/raw" / label
        folder.mkdir(parents=True)
        for i in range(10):
            Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(
                folder / f"image_{i}.png"
            )
    build_manifest(c)
    return c


@pytest.fixture
def locked(dataset):
    split_dataset(dataset, allow_fallback_groups=True, allow_small_test_set=True)
    return dataset
