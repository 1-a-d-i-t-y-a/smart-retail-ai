"""MobileNetV2 transfer-learning product classifier."""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np

# TensorFlow import is deferred so lightweight consumers (e.g. sentiment tests)
# don't pay the import cost.


def build_model(num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)):
    """Return a compiled MobileNetV2-based classifier with a frozen backbone."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    data_dir: str,
    model_out: str = "models/product_classifier_model.h5",
    labels_out: str = "models/product_labels.json",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 8,
    epochs: int = 5,
):
    """Train on an ImageFolder-style directory (`data_dir/<class>/*.jpg`)."""
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.1,
    )
    train_gen = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, subset="training", class_mode="categorical"
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size, subset="validation", class_mode="categorical"
    )

    model = build_model(num_classes=train_gen.num_classes, input_shape=(*img_size, 3))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_out, save_best_only=True),
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    os.makedirs(os.path.dirname(labels_out), exist_ok=True)
    labels = {v: k for k, v in train_gen.class_indices.items()}
    with open(labels_out, "w") as f:
        json.dump(labels, f)
    return model, history, labels


def predict_image(
    img_path: str,
    model_path: str = "models/product_classifier_model.h5",
    labels_path: str = "models/product_labels.json",
    img_size: Tuple[int, int] = (224, 224),
) -> Tuple[str, float]:
    """Predict class + confidence for a single image."""
    import tensorflow as tf

    with open(labels_path) as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])
