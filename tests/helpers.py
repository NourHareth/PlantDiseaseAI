"""Shared test data and utilities."""

import io

from PIL import Image

# The class folder names of the PlantVillage tomato dataset, as Keras sees them.
PLANTVILLAGE_FOLDERS = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def png_bytes(image: Image.Image, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **save_kwargs)
    return buffer.getvalue()


def jpeg_bytes(image: Image.Image, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", **save_kwargs)
    return buffer.getvalue()
