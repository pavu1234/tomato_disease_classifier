import shutil
from src.manifest import build_manifest
from src.duplicate_detection import find_duplicates, PAIR_COLUMNS, components
from src.config_loader import path_for


def test_exact_and_near_format(dataset):
    p = path_for(dataset, "raw_data_dir") / "healthy"
    shutil.copy2(p / "image_0.png", p / "copy.png")
    df = build_manifest(dataset)
    exact, near = find_duplicates(df)
    assert len(exact) == 1
    assert list(near.columns) == PAIR_COLUMNS
    pair = exact.iloc[0]
    assert pair.kind == "exact" and not pair.label_conflict


def test_near_duplicate_and_transitive_components(dataset):
    df = build_manifest(dataset).iloc[:3].copy()
    df["perceptual_hash"] = ["0000000000000000", "0000000000000001", "0000000000000003"]
    e, n = find_duplicates(df, 1)
    assert len(n) == 2
    assert n.review.eq("manual review required").all()
    assert len(set(components(df, e, n).values())) == 1


def test_label_conflict(dataset):
    raw = path_for(dataset, "raw_data_dir")
    shutil.copy2(raw / "healthy/image_0.png", raw / "early_blight/conflict.png")
    df = build_manifest(dataset)
    e, n = find_duplicates(df)
    assert e.label_conflict.any()
