import importlib
import sys
import pandas as pd

from tests.model_helpers import (
    install_fake_datasets_load_from_disk,
    install_fake_textblob,
    capture_to_csv_calls,
)

def test_yelp_model_smoke(monkeypatch):
    df = pd.DataFrame(
        {
            "label": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
            "text": ["Great!", "Terrible!", "Nice", "Bad", "Love it", "Hate it", "Good", "Awful", "Perfect", "Worst"],
        }
    )

    install_fake_datasets_load_from_disk(monkeypatch, df)
    install_fake_textblob(monkeypatch, polarity=0.1, subjectivity=0.2)
    csv_paths = capture_to_csv_calls(monkeypatch)

    module_name = "scripts.yelp_model"

    if module_name in sys.modules:
        del sys.modules[module_name]

    importlib.import_module(module_name)

    assert "datasets/processed/yelp_features.csv" in csv_paths
    assert "datasets/csvs/yelp_reviews.csv" in csv_paths
    assert "coefs/yelp_coefs.csv" in csv_paths

