import pandas as pd

from tests.model_helpers import (
    install_fake_datasets_load_from_disk,
    install_fake_textblob,
    capture_to_csv_calls,
    run_script_by_path,
)

def test_steam_model_smoke(monkeypatch):
    df = pd.DataFrame(
        {
            "playtime": [10, 5, 20, 1, 15, 2, 30, 3, 8, 4],
            "helpfulness": [1, 0, 2, 0, 3, 1, 5, 0, 1, 0],
            "review": ["Great game!", "Not good", "Loved it", "Awful", "Amazing", "Terrible", "Best", "Worst", "Fun", "Boring"],
            "post_date": ["2024-01-01"] * 10,
            "recommend": ["Recommended", "Not Recommended"] * 5,
            "early_access_review": [True, False] * 5,
        }
    )

    install_fake_datasets_load_from_disk(monkeypatch, df)
    install_fake_textblob(monkeypatch, polarity=0.05, subjectivity=0.1)
    csv_paths = capture_to_csv_calls(monkeypatch)

    run_script_by_path("scripts/steam_model.py", "steam_model_smoke")

    assert "datasets/processed/steam_features.csv" in csv_paths
    assert "datasets/csvs/steam_reviews.csv" in csv_paths
    assert "coefs/steam_coefs.csv" in csv_paths