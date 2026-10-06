import tempfile
from pathlib import Path
import numpy as np
from .config_loader import path_for
from .dataset_loader import load_dataset
from .evaluate import frozen_artifact
from .split_dataset import verify_lock
from .reporting import save_report


def export(c):
    import tensorflow as tf

    verify_lock(c, scan_test=False)
    model_path, _ = frozen_artifact(c)
    model = tf.keras.models.load_model(model_path)
    # Strip training-only augmentation and dropout by reusing the learned layers.
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="rgb_0_255")
    x = inputs
    for layer in model.layers[1:]:
        if layer.name == "training_augmentation" or isinstance(
            layer, tf.keras.layers.Dropout
        ):
            continue
        x = layer(x, training=False) if isinstance(layer, tf.keras.Model) else layer(x)
    serving = tf.keras.Model(inputs, x)
    val, names, _ = load_dataset(c, "val")
    samples = []
    counts = {i: 0 for i in range(len(names))}
    for images, labels in val:
        for img, label in zip(images.numpy(), labels.numpy()):
            label = int(label)
            if counts[label] < 10:
                samples.append(img)
                counts[label] += 1
        if all(n >= 10 for n in counts.values()):
            break
    if not samples:
        raise ValueError("Validation images required for export consistency check")
    samples = np.array(samples, dtype=np.float32)
    reference = model(samples, training=False).numpy()
    if not np.allclose(reference, serving(samples, training=False).numpy(), atol=1e-5):
        raise ValueError("Inference graph differs from Keras model")
    result = {
        "architecture": c["model"]["architecture"],
        "class_order": names,
        "sample_count": len(samples),
        "sample_counts_by_class": {names[i]: n for i, n in counts.items()},
        "sample_selection": "First up to 10 validation images per class in deterministic filename order",
        "comparison_split": "validation only",
        "required_preprocessing": "EXIF transpose, RGB, float32 [0,255], bilinear resize to model height/width. Normalization is embedded. Do not divide pixels by 255.",
        "input_shape": list(model.input_shape),
        "deployment_note": "On Raspberry Pi use a compatible TensorFlow Lite interpreter, allocate tensors, feed NHWC float32, and ship class_names.json. Benchmark on the actual Pi; no hardware latency is claimed.",
        "exports": {},
    }
    prefix = (
        "mobilenetv3"
        if c["model"]["architecture"] == "MobileNetV3Small"
        else "mobilenetv2"
    )
    with tempfile.TemporaryDirectory() as temp:
        # Export inference-only SavedModel avoids Keras 3 direct-converter variable issues.
        serving.export(temp)
        for mode in ["float32", "dynamic_range"]:
            converter = tf.lite.TFLiteConverter.from_saved_model(temp)
            if mode == "dynamic_range":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            blob = converter.convert()
            out = path_for(c, "models_dir") / f"tomato_disease_{prefix}_{mode}.tflite"
            out.write_bytes(blob)
            interpreter = tf.lite.Interpreter(model_content=blob)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            output = interpreter.get_output_details()[0]
            preds = []
            for sample in samples:
                interpreter.set_tensor(inp["index"], sample[None, ...])
                interpreter.invoke()
                preds.append(interpreter.get_tensor(output["index"])[0])
            preds = np.array(preds)
            agreement = float(np.mean(preds.argmax(1) == reference.argmax(1)))
            result["exports"][mode] = dict(
                path=str(out),
                size_bytes=len(blob),
                compression_vs_keras_percent=100
                * (1 - len(blob) / model_path.stat().st_size),
                top1_agreement=agreement,
                max_absolute_probability_difference=float(
                    np.max(np.abs(preds - reference))
                ),
                review_required=bool(agreement < 1.0),
                note="Agreement on a validation subset is not deployment accuracy; Keras archive may include optimizer state.",
            )
    a = result["exports"]
    a["dynamic_range"]["compression_vs_float32_percent"] = 100 * (
        1 - a["dynamic_range"]["size_bytes"] / a["float32"]["size_bytes"]
    )
    save_report(path_for(c, "reports_dir"), "tflite_export_report", result)
    return result
