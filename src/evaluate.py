import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from .config_loader import path_for
from .dataset_loader import load_dataset
from .split_dataset import verify_lock, read_lock
from .reporting import save_report, confusion_plots
from .utils import read_json, write_json, sha256, digest, now, LOG

FINAL_WARNING = """FINAL EVALUATION WARNING:
You are evaluating the frozen selected model on the locked test set.
Do not use these results to continue tuning the current model.
If you change architecture, data preprocessing, augmentation, thresholds, or hyperparameters after seeing these results, create a fresh locked test split for the next final evaluation.
A fresh split must use genuinely unseen independent test sources, not a reshuffle of exposed data."""
HONEST_NOTE = "The model achieved the reported score on a locked internal test set. This result does not establish field or drone deployment performance unless the test images were independently collected under representative field/drone conditions."


def predict_dataset(model, ds):
    ys = []
    ps = []
    times = []
    for x, y in ds:
        start = time.perf_counter()
        p = model(x, training=False).numpy()
        elapsed = time.perf_counter() - start
        ys.extend(y.numpy().tolist())
        ps.append(p)
        times.extend([elapsed * 1000 / len(p)] * len(p))
    if not ps:
        raise ValueError("Empty evaluation dataset")
    return np.array(ys), np.concatenate(ps), times


def compute_metrics(y, p, names):
    if p.shape != (len(y), len(names)) or not np.isfinite(p).all():
        raise ValueError("Invalid model output")
    pred = p.argmax(axis=1)
    report = classification_report(
        y,
        pred,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    metrics = dict(
        loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)).mean()),
        accuracy=float(accuracy_score(y, pred)),
        macro_precision=report["macro avg"]["precision"],
        macro_recall=report["macro avg"]["recall"],
        macro_f1=report["macro avg"]["f1-score"],
        weighted_precision=report["weighted avg"]["precision"],
        weighted_recall=report["weighted avg"]["recall"],
        weighted_f1=report["weighted avg"]["f1-score"],
        sample_count=len(y),
        class_metrics={n: report[n] for n in names},
    )
    warnings = []
    if metrics["accuracy"] < 0.75:
        warnings.append("Accuracy below 0.75")
    if metrics["macro_f1"] < 0.70:
        warnings.append("Macro F1 below 0.70")
    if any(report[n]["recall"] < 0.60 for n in names):
        warnings.append("At least one class recall below 0.60")
    if np.bincount(pred, minlength=len(names)).max() / len(pred) > 0.80:
        warnings.append("One class dominates predictions (>80%)")
    if min(report[n]["support"] for n in names) < 20:
        warnings.append(
            "Fewer than 20 images in at least one class: statistically unstable, preliminary"
        )
    metrics["warnings"] = warnings
    return metrics, report, pred


def frozen_artifact(c):
    model_path = path_for(c, "models_dir") / "best_model.keras"
    frozen = read_json(path_for(c, "models_dir") / "frozen_model.json")
    if frozen["model_sha256"] != sha256(model_path):
        raise ValueError("Selected model changed after freeze")
    if frozen["class_names_sha256"] != sha256(
        path_for(c, "models_dir") / "class_names.json"
    ):
        raise ValueError("Class order changed after freeze")
    lock = read_lock(c)
    if frozen["lock_checksum"] != lock["manifest_checksum"]:
        raise ValueError("Model belongs to a different split")
    if any(
        frozen["development_config"][k] != c[k] for k in ["classes", "image", "model"]
    ):
        raise ValueError(
            "Model/preprocessing config changed after freeze. Restore the saved training config"
        )
    return model_path, frozen


def development_config(c):
    return {k: c[k] for k in ["seed", "classes", "image", "model", "training"]}


def evaluate(c, split="val", final_evaluation=False):
    if split == "test_locked" and not final_evaluation:
        raise PermissionError(
            "Test data is reserved for final performance measurement. Pass --final-evaluation only after model development is complete"
        )
    if split not in ["val", "test_locked"]:
        raise ValueError("Evaluation supports val or test_locked only")
    verify_lock(c, scan_test=split == "test_locked")
    model_path, frozen = frozen_artifact(c)
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    if split == "test_locked":
        print(FINAL_WARNING, flush=True)
        exposure = path_for(c, "test_lock_manifest").parent / "final_exposure.json"
        if exposure.exists():
            old = read_json(exposure)
            if (
                old["model_sha256"] != frozen["model_sha256"]
                or old["lock_checksum"] != frozen["lock_checksum"]
            ):
                raise ValueError(
                    "Final holdout was exposed to another model. Refusing reuse"
                )
        else:
            # Record exposure BEFORE any test images are decoded; crashes remain fail-closed.
            write_json(
                exposure,
                dict(
                    timestamp=now(),
                    model_sha256=frozen["model_sha256"],
                    lock_checksum=frozen["lock_checksum"],
                    status="evaluation_started",
                ),
            )
    ds, names, paths = load_dataset(c, split, final_evaluation=final_evaluation)
    if model.output_shape[-1] != len(names):
        raise ValueError("Model output and class order mismatch")
    y, p, times = predict_dataset(model, ds)
    metrics, report, pred = compute_metrics(y, p, names)
    prefix = "final_test" if split == "test_locked" else "validation"
    r = path_for(c, "reports_dir")
    r.mkdir(parents=True, exist_ok=True)
    metrics["confusion_matrix"] = confusion_plots(y, pred, names, r, prefix)
    cm = np.array(metrics["confusion_matrix"])
    metrics["normalized_confusion_matrix"] = (
        cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ).tolist()
    metrics["inference_time"] = dict(
        mean_ms_per_image=float(np.mean(times)),
        median_ms_per_image=float(np.median(times)),
        p95_ms_per_image=float(np.percentile(times, 95)),
        note="Batched forward-pass wall time divided by batch size; includes first trace, excludes file decoding. Not Raspberry Pi latency.",
    )
    metrics["model_sha256"] = sha256(model_path)
    metrics["timestamp"] = now()
    lock = read_lock(c)
    metrics["performance_scope"] = (
        "preliminary internal performance"
        if lock["preliminary"]
        else (
            "held-out internal test performance"
            if split == "test_locked"
            else "validation development performance"
        )
    )
    write_json(r / (prefix + "_metrics.json"), metrics)
    write_json(r / (prefix + "_classification_report.json"), report)
    if split == "test_locked":
        lookup = {
            str((Path(c["root"]) / x["processed_path"]).resolve()): x
            for x in lock["test_images"]
        }
        rows = []
        for i, path in enumerate(paths):
            item = lookup[str(Path(path).resolve())]
            rows.append(
                dict(
                    image_path=item["processed_path"],
                    image_id=item["image_id"],
                    true_label=names[int(y[i])],
                    predicted_label=names[int(pred[i])],
                    confidence=float(p[i, pred[i]]),
                    probability_for_each_class=__import__("json").dumps(
                        {n: float(p[i, j]) for j, n in enumerate(names)}
                    ),
                    correct=bool(y[i] == pred[i]),
                    source_group=item["source_group"],
                    **{"probability_" + n: float(p[i, j]) for j, n in enumerate(names)}
                )
            )
        pd.DataFrame(rows).to_csv(r / "final_test_predictions.csv", index=False)
        note = HONEST_NOTE
        if min(report[n]["support"] for n in names) < 20:
            note += "\n\nWARNING: The test set contains fewer than 20 images per class. The result is statistically unstable and should be presented as preliminary."
        if lock["preliminary"]:
            note += "\n\nReport this as preliminary internal performance; inspect split limitations and source metadata."
        save_report(r, "final_test_summary", metrics, note)
    for w in metrics["warnings"]:
        LOG.warning(w)
    return metrics
