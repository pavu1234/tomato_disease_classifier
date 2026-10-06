import shutil
import time
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from .config_loader import path_for
from .utils import read_json, write_json, sha256, now, require_not_exposed, LOG
from .split_dataset import verify_lock, read_lock
from .dataset_loader import load_training_data
from .evaluate import predict_dataset, compute_metrics, development_config
from .reporting import save_report, history_plots


def train(c, stage="all", skip_fine_tune=False, resume=False):
    require_not_exposed(c)
    # No holdout bytes opened: signed-by-checksum inventory + prior full verification stat receipt.
    verify_lock(c, scan_test=False)
    import tensorflow as tf
    from .model import build_model, compile_model, fine_tune

    tf.keras.utils.set_random_seed(c["seed"])
    tf.config.experimental.enable_op_determinism()
    models = path_for(c, "models_dir")
    logs = path_for(c, "logs_dir")
    reports = path_for(c, "reports_dir")
    for p in [models, logs, reports]:
        p.mkdir(parents=True, exist_ok=True)
    statepath = logs / "training_state.json"
    if not resume and (statepath.exists() or (models / "best_model.keras").exists()):
        raise ValueError(
            "Run exists. Use --resume after interruption, or a new experiment directory for new development"
        )
    started = time.perf_counter()
    lock = read_lock(c)
    skip = skip_fine_tune or stage == "stage1"
    if resume:
        state = read_json(statepath)
        if (
            state["lock_checksum"] != lock["manifest_checksum"]
            or state["config"] != development_config(c)
            or state["skip_fine_tune"] != skip
        ):
            raise ValueError(
                "Resume requires the original configuration, stages, and dataset lock"
            )
        if state["status"] == "complete":
            raise ValueError("Training is already complete. Selected model is frozen")
    else:
        state = dict(
            status="training",
            stage="stage1",
            next_epoch=0,
            best_loss=None,
            config=development_config(c),
            skip_fine_tune=skip,
            lock_checksum=lock["manifest_checksum"],
            created_at=now(),
        )
    train_ds, val_ds, names = load_training_data(c)
    rows = [x for x in lock["all_assignments"] if x["split"] == "train"]
    y = np.array([names.index(x["class_label"]) for x in rows])
    counts = np.bincount(y, minlength=len(names))
    weights = None
    if counts.max() / counts.min() > c["training"]["class_imbalance_ratio_threshold"]:
        weights = {
            i: float(w)
            for i, w in enumerate(
                compute_class_weight(
                    class_weight="balanced", classes=np.arange(len(names)), y=y
                )
            )
        }
    write_json(
        logs / "class_weights.json",
        dict(
            used=weights is not None,
            weights=weights,
            training_counts={n: int(counts[i]) for i, n in enumerate(names)},
        ),
    )
    write_json(logs / "training_config.json", c)
    print(
        "Class counts:",
        lock["counts"],
        "\nClass order:",
        names,
        "\nClass weights:",
        weights,
        flush=True,
    )
    print(
        "Training will use TRAIN and VALIDATION sets only. The locked test set will not be loaded.",
        flush=True,
    )
    if resume:
        model = tf.keras.models.load_model(models / "last_model.keras")
        base_name = state["base_name"]
    else:
        model, base_name = build_model(c, len(names))
        state["base_name"] = base_name
        compile_model(model, c["training"]["stage1_learning_rate"])
        model.save(models / "last_model.keras")
        write_json(statepath, state)
    stage_results = state.get("stage_results", {})

    def fit_stage(label, model, epochs):
        bestpath = models / (
            "stage1_best_model.keras" if label == "stage1" else "best_model.keras"
        )
        history = logs / (
            "training_history_stage1.csv"
            if label == "stage1"
            else "training_history_finetune.csv"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            str(bestpath), monitor="val_loss", mode="min", save_best_only=True
        )
        if state.get("best_loss") is not None:
            checkpoint.best = state["best_loss"]

        class SaveState(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                value = float(logs["val_loss"])
                if not np.isfinite(value):
                    raise RuntimeError("Nonfinite validation loss; training stopped")
                state.update(
                    stage=label,
                    next_epoch=epoch + 1,
                    best_loss=(
                        min(value, state["best_loss"])
                        if state.get("best_loss") is not None
                        else value
                    ),
                )
                self.model.save(models / "last_model.keras")
                write_json(statepath, state)
                if (epoch + 1) % 3 == 0:
                    from .checkpointing import snapshot_project
                    snapshot_project(c["root"], label, epoch + 1)

        callbacks = [
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=c["training"]["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6
            ),
            tf.keras.callbacks.CSVLogger(str(history), append=state["next_epoch"] > 0),
            tf.keras.callbacks.TerminateOnNaN(),
            SaveState(),
        ]
        if state["next_epoch"] < epochs:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                initial_epoch=state["next_epoch"],
                callbacks=callbacks,
                class_weight=weights,
                verbose=2,
            )
        if not bestpath.exists():
            raise ValueError("No valid best checkpoint saved")
        best = tf.keras.models.load_model(bestpath)
        yy, pp, _ = predict_dataset(best, val_ds)
        metrics, _, _ = compute_metrics(yy, pp, names)
        stage_results[label] = metrics
        state["stage_results"] = stage_results
        return best

    if state["stage"] == "stage1":
        model = fit_stage("stage1", model, c["training"]["stage1_epochs"])
        if not skip:
            fine_tune(model, base_name, c["model"]["fine_tune_last_n_layers"])
            compile_model(model, c["training"]["fine_tune_learning_rate"])
            state.update(stage="finetune", next_epoch=0, best_loss=None)
            model.save(models / "last_model.keras")
            write_json(statepath, state)
    if not skip:
        model = fit_stage("finetune", model, c["training"]["fine_tune_epochs"])
    first = models / "stage1_best_model.keras"
    best = models / "best_model.keras"
    # Lower validation loss wins; macro F1 breaks exact loss ties. No test metrics.
    selected = min(
        stage_results,
        key=lambda s: (stage_results[s]["loss"], -stage_results[s]["macro_f1"]),
    )
    if selected == "stage1":
        shutil.copy2(first, best)
    model = tf.keras.models.load_model(best)
    write_json(reports / "validation_metrics.json", stage_results[selected])
    write_json(
        models / "frozen_model.json",
        dict(
            timestamp=now(),
            model_sha256=sha256(best),
            class_names_sha256=sha256(models / "class_names.json"),
            lock_checksum=lock["manifest_checksum"],
            development_config=development_config(c),
            selected_stage=selected,
        ),
    )
    state["status"] = "complete"
    write_json(statepath, state)
    history_plots(logs, reports)
    summary = dict(
        dataset_counts=lock["counts"],
        class_order=names,
        architecture=c["model"]["architecture"],
        pretraining="ImageNet",
        stage_results=stage_results,
        selected_stage=selected,
        parameter_count=int(model.count_params()),
        approximate_float32_weight_bytes=int(model.count_params() * 4),
        saved_model_bytes=best.stat().st_size,
        training_duration_this_invocation_seconds=time.perf_counter() - started,
        warnings=[
            "Limited/curated data cannot establish field generalization.",
            "Resume restores optimizer/epoch/best checkpoint but restarts callback patience and shuffle stream; not bitwise equivalent to uninterrupted training.",
        ],
        statement="No locked test-set metrics were used during model training or selection.",
    )
    save_report(reports, "training_summary", summary, summary["statement"])
    return summary
