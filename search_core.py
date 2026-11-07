
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)

MODEL_SIZE = 96


# 1. Load the model
def load_model():
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(MODEL_SIZE, MODEL_SIZE, 3)
    )
    return model


# 2. Prepare images for the model
def to_model_input(images_28, invert=True):

    x = np.asarray(images_28, dtype="float32")
    x = x.reshape(-1, 28, 28, 1)

    if invert:
        x = 255.0 - x

    x = tf.image.resize(
        x,
        (MODEL_SIZE, MODEL_SIZE)
    ).numpy()

    x = np.repeat(x, 3, axis=-1)

    return preprocess_input(x)


# 3. Convert images into vectors
def embed(model, images_28, invert=True, batch_size=64):

    batches = []

    for i in range(0, len(images_28), batch_size):

        batch = images_28[i:i + batch_size]

        vectors = model.predict(
            to_model_input(batch, invert),
            verbose=0
        )

        batches.append(vectors)

    vectors = np.concatenate(batches)

    vectors = tf.math.l2_normalize(
        vectors,
        axis=1
    ).numpy()

    return vectors


# 4. Prepare the user's uploaded image
def preprocess_query(image):

    # Convert image to grayscale
    image = image.convert("RGBA")

    white = Image.new("RGBA", image.size, "white")

    image = Image.alpha_composite(
        white,
        image
    )

    gray = np.asarray(
        image.convert("L"),
        dtype="float32"
    )

    # Check if the image is blank
    low = gray.min()
    high = gray.max()

    if high - low < 40:
        return None

    # Detect dark drawing strokes
    mask = gray < (low + high) / 2

    strokes = Image.fromarray(
        mask.astype("uint8") * 255
    )

    # Find drawing boundaries
    bbox = strokes.getbbox()

    if bbox is None:
        return None

    # Crop the drawing
    strokes = strokes.crop(bbox)

    w, h = strokes.size

    # Make the drawing square
    side = max(w, h)

    margin = int(side * 0.15) + 1

    size = side + (2 * margin)

    square = Image.new(
        "L",
        (size, size),
        0
    )

    left = (size - w) // 2
    top = (size - h) // 2

    square.paste(
        strokes,
        (left, top)
    )

    # Resize to 28x28
    small = square.resize(
        (28, 28),
        Image.Resampling.BOX
    )

    return np.asarray(
        small,
        dtype="uint8"
    )


# 5. Search for similar drawings
def search(query_vector, embeddings, k=25):

    scores = embeddings @ query_vector

    best = np.argsort(-scores)[:k]

    return best, scores[best]
