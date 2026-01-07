import pandas as pd
from tests.model_helpers import run_script_by_path

def test_figures_script_smoke(monkeypatch):

    SCRIPT_PATH = "scripts/create_figures.py"

    def fake_read_csv(path, *args, **kwargs):
        if path.startswith("coefs/") and path.endswith("_coefs.csv"):
            return pd.DataFrame(
                {
                    "feature": [
                        "num_scaler__sentiment_polarity",
                        "verified_ohe__verified_purchase_True",
                        "num_scaler__review_length_words",
                    ],
                    "coef": [0.4, -0.2, 0.1],
                }
            )

        if path.startswith("datasets/processed/") and path.endswith("_features.csv"):
            if "steam_features.csv" in path:
                return pd.DataFrame(
                    {
                        "sentiment_polarity": [0.2, -0.6, -0.1, 0.1],
                        "negative_experience": [1, 1, 0, 0],
                        "recommend": ["Recommended", "Not Recommended", "Recommended", "Not Recommended"],
                    }
                )

            if "yelp_features.csv" in path:
                return pd.DataFrame(
                    {
                        "sentiment_polarity": [0.1, -0.4, -0.2, 0.3],
                        "negative_experience": [1, 1, 0, 0],
                        "label": [0, 4, 1, 3],
                    }
                )

            return pd.DataFrame(
                {
                    "sentiment_polarity": [0.3, -0.5, -0.2, 0.1],
                    "negative_experience": [1, 1, 0, 0],
                    "rating": [5, 1, 2, 4],
                }
            )

        raise AssertionError(f"Unexpected pd.read_csv path in test: {path}")

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    monkeypatch.setattr("os.path.exists", lambda p: True, raising=True)

    saved = []

    def fake_savefig(path, *args, **kwargs):
        saved.append(path)

    monkeypatch.setattr("matplotlib.pyplot.savefig", fake_savefig, raising=True)

    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None, raising=True)

    run_script_by_path(SCRIPT_PATH)

    # h2 is always called, so at minimum this should exist:
    assert any(p.endswith("h2_cross_platform_violin.png") for p in saved)

    # h3 creates 3 plots
    assert any(p.endswith("h3_amazon_beauty_sentiment_vs_group.png") for p in saved)
    assert any(p.endswith("h3_yelp_sentiment_vs_group.png") for p in saved)
    assert any(p.endswith("h3_steam_sentiment_vs_group.png") for p in saved)

    # h1 creates one per coef file key
    assert any(p.endswith("h1_amazon_beauty_top_coefficients.png") for p in saved)
    assert any(p.endswith("h1_steam_top_coefficients.png") for p in saved)
    assert any(p.endswith("h1_yelp_top_coefficients.png") for p in saved)