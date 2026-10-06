"""PNG processed datasets. Conversion in split handles WEBP, EXIF and RGB once.
Models accept RGB float32 [0,255]; architecture normalization is inside the model.
"""

from .config_loader import path_for
from .utils import write_json, read_json


def load_dataset(c, split, final_evaluation=False, save_names=False):
    if split not in ("train", "val", "test_locked"):
        raise ValueError("Invalid dataset split")
    if split == "test_locked" and not final_evaluation:
        raise PermissionError(
            "Locked test data requires --final-evaluation; use --split val for development"
        )
    import tensorflow as tf

    key = {"train": "train_dir", "val": "val_dir", "test_locked": "locked_test_dir"}[
        split
    ]
    ds = tf.keras.utils.image_dataset_from_directory(
        path_for(c, key),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=None,
        image_size=(c["image"]["height"], c["image"]["width"]),
        shuffle=False,
        interpolation="bilinear",
    )
    names = list(ds.class_names)
    paths = list(ds.file_paths)
    if set(names) != set(c["classes"]["expected"]):
        raise ValueError("Unexpected/missing classes")
    dest = path_for(c, "models_dir") / "class_names.json"
    if save_names:
        if split != "train":
            raise ValueError("Only training loader may save class order")
        write_json(dest, names)
    elif names != read_json(dest):
        raise ValueError("Loader class order differs from saved class names")
    if c["training"]["cache_dataset"]:
        ds = ds.cache()
    if split == "train":
        ds = ds.shuffle(len(paths), seed=c["seed"], reshuffle_each_iteration=True)
    ds = ds.batch(c["training"]["batch_size"])
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options).prefetch(tf.data.AUTOTUNE)
    return ds, names, paths


def load_training_data(c):
    train, names, _ = load_dataset(c, "train", save_names=True)
    val, val_names, _ = load_dataset(c, "val")
    if names != val_names:
        raise ValueError("Class order mismatch")
    return train, val, names
