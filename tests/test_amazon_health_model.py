import pandas as pd

from tests.model_helpers import (
    install_fake_datasets_load_from_disk,
    install_fake_textblob,
    capture_to_csv_calls,
    run_script_by_path,
)


def test_amazon_health_model_smoke(monkeypatch):
    df = pd.DataFrame(
        {
            "rating": [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
            "text": [
                "Great!", "Terrible!", "Nice", "Bad", "Love it",
                "Hate it", "Good", "Awful", "Perfect", "Worst"
            ],
            "helpful_vote": [0, 1, 0, 2, 1, 0, 3, 1, 0, 2],
            "verified_purchase": [True, False, True, False, True, False, True, False, True, False],
            "timestamp": [1700000000000] * 10,
        }
    )

    install_fake_datasets_load_from_disk(monkeypatch, df)
    install_fake_textblob(monkeypatch, polarity=0.1, subjectivity=0.2)
    csv_paths = capture_to_csv_calls(monkeypatch)

    run_script_by_path("scripts/amazon_health_model.py")

    assert "datasets/processed/amazon_health_features.csv" in csv_paths
    assert "datasets/csvs/amazon_health.csv" in csv_paths
    assert "coefs/amazon_health_coefs.csv" in csv_paths

    assert csv_paths, "No CSV writes were captured; script likely did not execute main()"
