import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .config_loader import path_for
from .utils import read_json, write_json, now

CONFIDENCE_NOTE = "Confidence indicates classifier certainty for this image; it does not measure infected leaf area or disease severity."


def prediction_payload(probabilities, names, elapsed, model_path, threshold):
    p = np.asarray(probabilities, dtype=float)
    if (
        p.shape != (len(names),)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.isclose(p.sum(), 1, atol=1e-3)
    ):
        raise ValueError("Invalid class probabilities")
    idx = int(p.argmax())
    confidence = float(p[idx])
    return dict(
        predicted_class=names[idx],
        confidence=confidence,
        class_probabilities={n: float(p[i]) for i, n in enumerate(names)},
        inference_time_milliseconds=float(elapsed),
        model_path=str(model_path),
        timestamp=now(),
        warning=(
            "Low confidence: seek expert review" if confidence < threshold else None
        ),
        confidence_note=CONFIDENCE_NOTE,
        scope_note="Closed-set classifier: always selects one of three classes. Cannot establish that an image is a tomato leaf or reject other diseases reliably.",
    )


def prepare_image(path, height, width):
    import tensorflow as tf

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.array(im)
    # Same tf.image.resize bilinear as image_dataset_from_directory, RGB [0,255].
    x = tf.image.resize(tf.cast(rgb, tf.float32), (height, width), method="bilinear")
    return x[None, ...], rgb


def predict(c, image_path, save_annotated=False):
    import tensorflow as tf

    model_path = path_for(c, "models_dir") / "best_model.keras"
    model = tf.keras.models.load_model(model_path)
    names = read_json(path_for(c, "models_dir") / "class_names.json")
    if len(names) != model.output_shape[-1] or set(names) != set(
        c["classes"]["expected"]
    ):
        raise ValueError("Saved class names do not match model output")
    frozen = read_json(path_for(c, "models_dir") / "frozen_model.json")
    from .utils import sha256

    if (
        sha256(model_path) != frozen["model_sha256"]
        or sha256(path_for(c, "models_dir") / "class_names.json")
        != frozen["class_names_sha256"]
    ):
        raise ValueError("Frozen artifact integrity failed")
    threshold = frozen["development_config"]["model"]["low_confidence_threshold"]
    x, rgb = prepare_image(
        image_path, int(model.input_shape[1]), int(model.input_shape[2])
    )
    start = time.perf_counter()
    p = model(x, training=False).numpy()[0]
    elapsed = (time.perf_counter() - start) * 1000
    result = prediction_payload(p, names, elapsed, model_path, threshold)
    dest = path_for(c, "logs_dir") / "predictions"
    dest.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    if save_annotated:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[1] < 480:
            bgr = cv2.resize(
                bgr, (480, max(1, round(bgr.shape[0] * 480 / bgr.shape[1])))
            )
        canvas = cv2.copyMakeBorder(
            bgr, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
        )
        cv2.putText(
            canvas,
            f"{result['predicted_class']}: {result['confidence']:.1%}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        if result["warning"]:
            cv2.putText(
                canvas,
                "LOW CONFIDENCE - expert review",
                (12, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1,
            )
        out = dest / (stem + ".png")
        if not cv2.imwrite(str(out), canvas):
            raise OSError("Could not save annotated image")
        result["annotated_image"] = str(out)
    write_json(dest / (stem + ".json"), result)
    return result
