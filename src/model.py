import tensorflow as tf


def build_model(c, n_classes):
    L = tf.keras.layers
    shape = (c["image"]["height"], c["image"]["width"], 3)
    inputs = L.Input(shape=shape, name="rgb_0_255")
    aug = tf.keras.Sequential(
        [
            L.RandomFlip("horizontal", seed=c["seed"]),
            L.RandomRotation(0.08, seed=c["seed"] + 1),
            L.RandomZoom(0.10, seed=c["seed"] + 2),
            L.RandomContrast(0.10, seed=c["seed"] + 3),
            L.RandomBrightness(0.08, value_range=(0, 255), seed=c["seed"] + 4),
        ],
        name="training_augmentation",
    )
    x = aug(inputs)
    try:
        if c["model"]["architecture"] == "MobileNetV3Small":
            base = tf.keras.applications.MobileNetV3Small(
                input_shape=shape,
                include_top=False,
                weights="imagenet",
                include_preprocessing=True,
            )
        else:
            # Exactly MobileNetV2 preprocess_input: RGB [0,255] -> [-1,1].
            x = L.Rescaling(1 / 127.5, offset=-1, name="mobilenetv2_preprocessing")(x)
            base = tf.keras.applications.MobileNetV2(
                input_shape=shape, include_top=False, weights="imagenet"
            )
    except Exception as e:
        raise RuntimeError(
            "ImageNet weights could not be loaded. Training aborted; random initialization is forbidden. Check network/cache."
        ) from e
    base.trainable = False
    x = base(
        x, training=False
    )  # BatchNorm always inference mode, even when fine-tuning.
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(c["model"]["dropout_1"])(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(c["model"]["dropout_2"])(x)
    return (
        tf.keras.Model(
            inputs,
            L.Dense(n_classes, activation="softmax", name="class_probabilities")(x),
        ),
        base.name,
    )


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def fine_tune(model, base_name, n):
    if n < 1:
        raise ValueError("fine_tune_last_n_layers must be >=1")
    base = model.get_layer(base_name)
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= len(base.layers) - n and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
