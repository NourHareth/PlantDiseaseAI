import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from PIL import Image

from plantdisease import config
from plantdisease.inference import PlantDiseaseClassifier, load_image
from plantdisease.knowledge import KnowledgeBase


@pytest.fixture(scope="session")
def classifier() -> PlantDiseaseClassifier:
    return PlantDiseaseClassifier.load()


@pytest.fixture(scope="session")
def knowledge() -> KnowledgeBase:
    return KnowledgeBase.from_json()


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    return load_image(config.SAMPLE_IMAGE_PATH)


@pytest.fixture
def synthetic_images() -> dict[str, Image.Image]:
    rng = np.random.default_rng(0)
    return {
        "black": Image.new("RGB", (300, 300), (0, 0, 0)),
        "white": Image.new("RGB", (300, 300), (255, 255, 255)),
        "green": Image.new("RGB", (300, 300), (0, 153, 0)),
        "noise": Image.fromarray(rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)),
        "grey_gradient": Image.fromarray(
            np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (300, 1))
        ).convert("RGB"),
    }
