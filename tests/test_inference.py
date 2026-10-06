import json
import numpy as np
import pytest
from src.inference import prediction_payload, CONFIDENCE_NOTE
from src.config_loader import path_for
from src.utils import read_json


def test_serializable_probabilities_and_note():
    result = prediction_payload(
        [0.2, 0.35, 0.45],
        ["early_blight", "healthy", "late_blight"],
        1.3,
        "model.keras",
        0.5,
    )
    assert json.loads(json.dumps(result))["predicted_class"] == "late_blight"
    assert result["warning"] and result["confidence_note"] == CONFIDENCE_NOTE
    assert (
        "does not measure infected leaf area or disease severity"
        in result["confidence_note"]
    )
    assert "severity" not in result


def test_invalid_probs():
    with pytest.raises(ValueError):
        prediction_payload([np.nan, 0.5, 0.5], ["a", "b", "c"], 1, "m", 0.5)


def test_loader_order_and_no_holdout_access(locked, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from src.dataset_loader import load_training_data

    original = tf.keras.utils.image_dataset_from_directory
    calls = []

    def guarded(directory, **kw):
        assert "test_locked" not in str(directory)
        assert kw["label_mode"] == "int"
        ds = original(directory, **kw)
        calls.append((str(directory), list(ds.class_names)))
        return ds

    monkeypatch.setattr(tf.keras.utils, "image_dataset_from_directory", guarded)
    train, val, names = load_training_data(locked)
    assert len(calls) == 2
    assert (
        read_json(path_for(locked, "models_dir") / "class_names.json")
        == calls[0][1]
        == names
    )
    x, y = next(iter(val))
    assert x.shape[1:] == (224, 224, 3)


def test_single_image_matches_loader(locked):
    pytest.importorskip("tensorflow")
    from src.dataset_loader import load_dataset
    from src.inference import prepare_image

    load_dataset(locked, "train", save_names=True)
    ds, names, paths = load_dataset(locked, "val")
    x, y = next(iter(ds))
    single, _ = prepare_image(paths[0], 224, 224)
    np.testing.assert_allclose(x[0].numpy(), single[0].numpy(), atol=1e-4)


def test_export_uses_validation_only():
    # Static policy regression plus executable converter smoke test below.
    import ast, inspect
    from src import export_tflite

    tree = ast.parse(inspect.getsource(export_tflite))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "load_dataset"
    ]
    assert len(calls) == 1 and calls[0].args[1].value == "val"


def test_end_to_end_contract_on_temporary_fixtures(locked, monkeypatch):
    """Tiny test-only network validates plumbing, NOT plant-disease performance."""
    tf = pytest.importorskip("tensorflow")
    from src import model as model_module
    from src.train import train
    from src.evaluate import evaluate
    from src.inference import predict
    from src.export_tflite import export
    from src.utils import sha256

    def fixture_model(c, n):
        base = tf.keras.Sequential(
            [
                tf.keras.layers.Input((32, 32, 3)),
                tf.keras.layers.Rescaling(1 / 255),
                tf.keras.layers.Conv2D(4, 3, activation="relu"),
                tf.keras.layers.BatchNormalization(),
            ],
            name="fixture_backbone",
        )
        x = tf.keras.Input((32, 32, 3))
        y = base(x, training=False)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        return (
            tf.keras.Model(x, tf.keras.layers.Dense(n, activation="softmax")(y)),
            base.name,
        )

    monkeypatch.setattr(model_module, "build_model", fixture_model)
    locked["image"].update(height=32, width=32)
    locked["training"].update(stage1_epochs=1, fine_tune_epochs=1, batch_size=4)
    raw = path_for(locked, "raw_data_dir") / "healthy/image_0.png"
    before = sha256(raw)
    train(locked)
    evaluate(locked, "val")
    # Final evaluation records exposure and permits identical-artifact reproducibility.
    evaluate(locked, "test_locked", True)
    assert (path_for(locked, "reports_dir") / "final_test_predictions.csv").exists()
    result = predict(locked, raw, save_annotated=True)
    assert result["predicted_class"] in locked["classes"]["expected"]
    export_result = export(locked)
    assert export_result["comparison_split"] == "validation only"
    assert export_result["exports"]["float32"]["top1_agreement"] == 1
    assert sha256(raw) == before
    with pytest.raises(RuntimeError, match="Final evaluation"):
        train(locked, resume=True)


def test_imagenet_failure_never_falls_back(dataset, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from src.model import build_model

    calls = []

    def unavailable(**kw):
        calls.append(kw)
        raise OSError("test: weights unavailable")

    monkeypatch.setattr(tf.keras.applications, "MobileNetV3Small", unavailable)
    with pytest.raises(RuntimeError, match="random initialization is forbidden"):
        build_model(dataset, 3)
    assert len(calls) == 1 and calls[0]["weights"] == "imagenet"
