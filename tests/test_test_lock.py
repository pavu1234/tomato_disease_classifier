from pathlib import Path
import builtins
import pytest
from src.config_loader import path_for
from src.split_dataset import verify_lock, read_lock, split_dataset
from src.utils import write_json


def test_changed_test_hash_detected(locked):
    row = read_lock(locked)["test_images"][0]
    p = Path(locked["root"]) / row["processed_path"]
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or missing"):
        verify_lock(locked)


def test_training_preflight_never_opens_test_images(locked, monkeypatch):
    testroot = path_for(locked, "locked_test_dir").resolve()
    orig = builtins.open

    def guarded(file, *args, **kw):
        if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(
            testroot
        ):
            raise AssertionError("Test image was opened by training preflight")
        return orig(file, *args, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    assert verify_lock(locked, scan_test=False)["passed"]


def test_train_validation_tampering(locked):
    row = next(x for x in read_lock(locked)["all_assignments"] if x["split"] == "train")
    (Path(locked["root"]) / row["processed_path"]).write_bytes(b"tamper")
    with pytest.raises(ValueError, match="Changed or missing"):
        verify_lock(locked, scan_test=False)


def test_test_change_invalidates_training_receipt(locked):
    p = Path(locked["root"]) / read_lock(locked)["test_images"][0]["processed_path"]
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Test changed"):
        verify_lock(locked, scan_test=False)


def test_added_file_detected(locked):
    p = path_for(locked, "locked_test_dir") / "healthy/extra.png"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory changed"):
        verify_lock(locked)


def test_final_flag_required_before_loading(dataset, monkeypatch):
    from src import evaluate

    monkeypatch.setattr(
        evaluate, "load_dataset", lambda *a, **k: pytest.fail("Loaded test data")
    )
    with pytest.raises(PermissionError, match="reserved"):
        evaluate.evaluate(dataset, "test_locked")
    from src.dataset_loader import load_dataset

    with pytest.raises(PermissionError):
        load_dataset(dataset, "test_locked")


def test_exposure_blocks_training_and_rebuild(locked):
    from src.train import train

    write_json(
        path_for(locked, "test_lock_manifest").parent / "final_exposure.json",
        {"exposed": True},
    )
    with pytest.raises(RuntimeError, match="Final evaluation"):
        train(locked)
    with pytest.raises(RuntimeError, match="Final evaluation"):
        split_dataset(locked, force_rebuild=True)
