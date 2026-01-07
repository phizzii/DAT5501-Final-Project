import pandas as pd
from tests.model_helpers import run_script_by_path

def test_key_figure_script_smoke(monkeypatch):
    SCRIPT_PATH = "scripts/create_key_figure.py"

    monkeypatch.setattr("os.path.exists", lambda p: True, raising=True)

    def fake_read_csv(path, *args, **kwargs):
        if path.startswith("coefs/") and path.endswith("_coefs.csv"):
            return pd.DataFrame(
                {
                    "feature": [
                        "num_scaler__sentiment_polarity",
                        "verified_ohe__verified_purchase_True",
                        "num_scaler__review_length_words",
                        "early_access_ohe__early_access_review_Early Access Review",
                        "num_scaler__exclamation_count",
                        "num_scaler__capital_ratio",
                        "num_scaler__avg_sentence_length",
                        "num_scaler__sentiment_subjectivity",
                        "num_scaler__review_length_chars",
                    ],
                    "coef": [0.4, -0.25, 0.15, -0.1, 0.08, -0.05, 0.03, 0.02, -0.01],
                }
            )
        raise AssertionError(f"Unexpected pd.read_csv path in test: {path}")

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None, raising=True)

    saved = {"path": None}

    def fake_savefig(path, *args, **kwargs):
        saved["path"] = path

    monkeypatch.setattr("matplotlib.pyplot.savefig", fake_savefig, raising=True)

    run_script_by_path(SCRIPT_PATH)

    assert saved["path"] is not None, "Expected a figure to be saved via plt.savefig"
    assert saved["path"].endswith("outputs/figures/key_figure_cross_platform_coefficients.png")
