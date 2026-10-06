from pathlib import Path
import pandas as pd
from src.manifest import build_manifest, read_manifest
from src.config_loader import path_for


def test_label_corrupt_zero_and_unsupported(dataset):
    raw = path_for(dataset, "raw_data_dir") / "healthy"
    (raw / "broken.jpg").write_bytes(b"not an image")
    (raw / "zero.png").write_bytes(b"")
    (raw / "readme.txt").write_text("hello")
    df = build_manifest(dataset)
    assert set(df.class_label) == set(dataset["classes"]["expected"])
    assert not df[
        df.filename.isin(["broken.jpg", "zero.png", "readme.txt"])
    ].is_valid_image.any()
    assert "corrupt" in df[df.filename == "broken.jpg"].iloc[0].audit_flags


def test_manual_metadata_preserved(dataset):
    p = path_for(dataset, "manifest_path")
    df = pd.read_csv(p, keep_default_na=False)
    df.loc[0, ["source_group", "source_type", "plant_id", "notes"]] = [
        "farmA-plant7",
        "verified_group",
        "7",
        "manual",
    ]
    df.to_csv(p, index=False)
    new = build_manifest(dataset)
    assert new.loc[0, "source_group"] == "farmA-plant7"
    assert new.loc[0, "notes"] == "manual"


def test_small_and_grayscale(dataset):
    from PIL import Image

    raw = path_for(dataset, "raw_data_dir") / "healthy"
    Image.new("RGB", (5, 5)).save(raw / "small.png")
    Image.new("L", (40, 40)).save(raw / "gray.png")
    df = build_manifest(dataset).set_index("filename")
    assert not df.loc["small.png", "is_valid_image"]
    assert df.loc["gray.png", "is_valid_image"]
    assert df.loc["gray.png", "image_mode"] == "L"


def test_derived_metadata_tampering(dataset):
    import pytest

    p = path_for(dataset, "manifest_path")
    df = pd.read_csv(p, keep_default_na=False)
    df.loc[0, "perceptual_hash"] = "0" * 16
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="metadata changed"):
        read_manifest(dataset)


def test_empty_dataset_audit_is_honest(dataset):
    import shutil
    from src.dataset_audit import audit

    shutil.rmtree(path_for(dataset, "raw_data_dir"))
    report = audit(dataset)
    assert report["total_files"] == 0
    assert not report["metadata_declared_sufficient"]
