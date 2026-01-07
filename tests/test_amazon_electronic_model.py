import importlib
import sys
import pandas as pd

from tests.model_helpers import (
    install_fake_datasets_load_from_disk,
    install_fake_textblob,
    capture_to_csv_calls,
)

def test_amazon_health_model_smoke(monkeypatch):
    df = pd.DataFrame(
        {
            "rating": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
            "text": ["Great!", "Terrible!", "Nice", "Bad", "Love it", "Hate it", "Good", "Awful", "Perfect", "Worst"],
            "helpful_vote": [0, 1, 0, 2, 1, 0, 3, 1, 0, 2],
            "verified_purchase": [True, False, True, False, True, False, True, False, True, False],
            "timestamp": [1700000000000] * 10,
        }
    )

    install_fake_datasets_load_from_disk(monkeypatch, df)
    install_fake_textblob(monkeypatch, polarity=0.1, subjectivity=0.2)
    csv_paths = capture_to_csv_calls(monkeypatch)

    module_name = "scripts.amazon_electronics_model"

    if module_name in sys.modules:
        del sys.modules[module_name]

    importlib.import_module(module_name)

    assert "datasets/processed/amazon_electronics_features.csv" in csv_paths
    assert "datasets/csvs/amazon_electronics.csv" in csv_paths
    assert "coefs/amazon_electronics_coefs.csv" in csv_paths

