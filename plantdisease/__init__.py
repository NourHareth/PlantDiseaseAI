"""PlantDisease AI: tomato leaf disease classification with MobileNetV2."""

import os

# Keep TensorFlow's C++ start-up logging out of the console. Must be set before
# TensorFlow is imported anywhere in the process.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "2.0.0"
