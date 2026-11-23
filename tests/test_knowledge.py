import pytest

from helpers import PLANTVILLAGE_FOLDERS
from plantdisease import config
from plantdisease.inference import load_class_names
from plantdisease.knowledge import CATEGORIES, normalize_label


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Tomato___Spider_mites Two-spotted_spider_mite", "spider mites two spotted spider mite"),
        ("Spider mites Two-spotted spider mite", "spider mites two spotted spider mite"),
        ("Tomato___healthy", "healthy"),
        ("Leaf Mold", "leaf mold"),
        ("  Target   Spot ", "target spot"),
        ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "tomato yellow leaf curl virus"),
    ],
)
def test_normalize_label(label, expected):
    assert normalize_label(label) == expected


def test_every_model_class_has_a_guide(knowledge):
    class_names = load_class_names(config.MODELS_DIR / config.CLASS_NAMES_FILENAME)
    assert knowledge.missing(class_names) == []


def test_raw_plantvillage_folder_names_resolve(knowledge):
    # A model retrained with scripts/train.py stores the raw folder names.
    assert knowledge.missing(PLANTVILLAGE_FOLDERS) == []


def test_entries_are_complete(knowledge):
    assert len(knowledge) == 10
    for folder in PLANTVILLAGE_FOLDERS:
        info = knowledge.get(folder)
        assert info.category in CATEGORIES and info.category != "unknown"
        assert info.display_name and info.description and info.cause
        assert len(info.symptoms) >= 2 and len(info.actions) >= 2


def test_unknown_label_gets_placeholder(knowledge):
    info = knowledge.get("Potato___Wart")
    assert info.category == "unknown"
    assert info.display_name == "Potato wart"
    assert info.actions


def test_healthy_flag(knowledge):
    assert knowledge.get("healthy").is_healthy
    assert not knowledge.get("Late blight").is_healthy
