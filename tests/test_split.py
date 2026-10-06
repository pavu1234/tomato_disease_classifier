import shutil
import pandas as pd
import pytest
from src.config_loader import path_for
from src.manifest import build_manifest
from src.split_dataset import split_dataset
from src.utils import sha256


def test_requires_metadata(dataset):
    with pytest.raises(ValueError, match="metadata is not verified"):
        split_dataset(dataset)


def test_groups_hashes_and_png_conversion(dataset):
    raw = path_for(dataset, "raw_data_dir")
    shutil.copy2(raw / "healthy/image_0.png", raw / "healthy/copy.png")
    df = build_manifest(dataset)
    # Real-looking metadata is test input, not asserted real evidence.
    df["source_group"] = ["source-" + str(i // 2) for i in range(len(df))]
    df["source_type"] = "verified_group"
    df.to_csv(path_for(dataset, "manifest_path"), index=False)
    split_dataset(dataset, allow_small_test_set=True)
    df = pd.read_csv(path_for(dataset, "reports_dir") / "split_manifest.csv")
    for key in ["source_group", "sha256_hash", "processed_sha256"]:
        assert df.groupby(key).split.nunique().max() == 1
    assert set(df.split) == {"train", "val", "test_locked"}
    assert set(df.groupby("split").class_label.nunique()) == {3}
    assert all(p.endswith(".png") for p in df.processed_path)


def test_missing_class_fails(dataset):
    shutil.rmtree(path_for(dataset, "raw_data_dir") / "healthy")
    build_manifest(dataset)
    with pytest.raises(ValueError, match="class is missing"):
        split_dataset(dataset, allow_fallback_groups=True)


def test_conflicts_fail(dataset):
    raw = path_for(dataset, "raw_data_dir")
    shutil.copy2(raw / "healthy/image_0.png", raw / "late_blight/conflict.png")
    build_manifest(dataset)
    with pytest.raises(ValueError, match="label-conflict"):
        split_dataset(dataset, allow_fallback_groups=True)


def test_small_set_requires_override(dataset):
    with pytest.raises(ValueError, match="No feasible"):
        split_dataset(dataset, allow_fallback_groups=True)


def test_dry_run_does_not_modify_lock_or_raw(locked):
    lock = path_for(locked, "test_lock_manifest")
    before = sha256(lock)
    manifest = path_for(locked, "reports_dir") / "split_manifest.csv"
    original = sha256(manifest)
    split_dataset(
        locked,
        dry_run=True,
        allow_fallback_groups=True,
        allow_small_test_set=True,
        seed=5,
    )
    assert sha256(lock) == before and sha256(manifest) == original
    with pytest.raises(ValueError, match="already exists"):
        split_dataset(locked, allow_fallback_groups=True)
