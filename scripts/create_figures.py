import os
import pandas as pd
import matplotlib.pyplot as plt

# hopefully fix label problem for steam. NOW IT DOES
def clean_feature_name(name: str) -> str:
    prefixes = [
        "num_scaler__",
        "remainder__",
        "verified_ohe__",
        "early_access_ohe__",
    ]
    for p in prefixes:
        name = name.replace(p, "")

    # special case: steam early access one-hot names
    # example raw: early_access_review_Early Access Review
    if name.startswith("early_access_review_"):
        return "Early Access Review"

    # general formatting
    name = name.replace("_", " ").strip()
    return name.title()

def trim_series(series: pd.Series, low=0.01, high=0.99) -> pd.Series:
    return series.clip(
        lower=series.quantile(low),
        upper=series.quantile(high)
    )


def dataset_name(key: str) -> str:
    names = {
        "amazon_beauty": "Amazon (Beauty)",
        "amazon_health": "Amazon (Health)",
        "amazon_electronics": "Amazon (Electronics)",
        "amazon_clothing": "Amazon (Clothing)",
        "steam": "Steam",
        "yelp": "Yelp"
    }
    return names.get(key, key)

# h1 — top coefficients

def plot_top_coefficients(coef_csv, dataset_key, out_dir, top_n=8):
    df = pd.read_csv(coef_csv)
    df["abs"] = df["coef"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n)

    df["feature"] = df["feature"].apply(clean_feature_name)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["coef"][::-1])
    plt.title(f"{dataset_name(dataset_key)} — Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient value")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"h1_{dataset_key}_top_coefficients.png"),
        dpi=200
    )
    plt.close()

# h2 — cross-platform sentiment distribution (negative only)

def plot_cross_platform_violin(feature_paths, out_dir):
    frames = []

    for key, path in feature_paths.items():
        df = pd.read_csv(path)
        if "sentiment_polarity" not in df.columns:
            continue
        if "negative_experience" not in df.columns:
            continue

        df = df[df["negative_experience"] == 1]
        df = df[["sentiment_polarity"]].copy()
        df["platform"] = dataset_name(key)
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    data = []
    labels = []

    for platform in sorted(all_data["platform"].unique()):
        s = all_data.loc[
            all_data["platform"] == platform,
            "sentiment_polarity"
        ]
        s = trim_series(s)
        data.append(s)
        labels.append(platform)

    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title("Sentiment Polarity Distribution Across Platforms (Negative Reviews)")
    plt.ylabel("Sentiment polarity")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, "h2_cross_platform_violin.png"),
        dpi=200
    )
    plt.close()

# h3 — sentiment vs outcome group (violin plots)

def plot_sentiment_vs_group(feature_csv, dataset_key, out_dir):
    df = pd.read_csv(feature_csv)

    if dataset_key.startswith("amazon"):
        df["group"] = df["rating"].round().astype(int)
        title = "Sentiment vs Rating"

    elif dataset_key == "yelp":
        df["group"] = df["label"] + 1
        title = "Sentiment vs Star Rating"

    elif dataset_key == "steam":
        df["group"] = df["recommend"]
        title = "Sentiment vs Recommendation"

    else:
        return

    data = []
    labels = []

    for g in sorted(df["group"].dropna().unique()):
        s = df.loc[df["group"] == g, "sentiment_polarity"]
        s = trim_series(s)
        data.append(s)
        labels.append(str(g))

    plt.figure(figsize=(10, 6))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title(f"{dataset_name(dataset_key)} — {title}")
    plt.ylabel("Sentiment polarity")
    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"h3_{dataset_key}_sentiment_vs_group.png"),
        dpi=200
    )
    plt.close()

# main

def main():
    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)

    coef_paths = {
        "amazon_beauty": "coefs/amazon_beauty_coefs.csv",
        "amazon_health": "coefs/amazon_health_coefs.csv",
        "amazon_electronics": "coefs/amazon_electronics_coefs.csv",
        "amazon_clothing": "coefs/amazon_clothing_coefs.csv",
        "steam": "coefs/steam_coefs.csv",
        "yelp": "coefs/yelp_coefs.csv",
    }

    feature_paths = {
        "amazon_beauty": "datasets/processed/amazon_beauty_features.csv",
        "amazon_health": "datasets/processed/amazon_health_features.csv",
        "amazon_electronics": "datasets/processed/amazon_electronics_features.csv",
        "amazon_clothing": "datasets/processed/amazon_clothing_features.csv",
        "steam": "datasets/processed/steam_features.csv",
        "yelp": "datasets/processed/yelp_features.csv",
    }

    # h1
    for key, path in coef_paths.items():
        if os.path.exists(path):
            plot_top_coefficients(path, key, out_dir)

    # h2 (one amazon dataset only cause there is many)
    plot_cross_platform_violin(
        {
            "amazon_beauty": feature_paths["amazon_beauty"],
            "steam": feature_paths["steam"],
            "yelp": feature_paths["yelp"],
        },
        out_dir
    )

    # h3
    plot_sentiment_vs_group(feature_paths["amazon_beauty"], "amazon_beauty", out_dir)
    plot_sentiment_vs_group(feature_paths["yelp"], "yelp", out_dir)
    plot_sentiment_vs_group(feature_paths["steam"], "steam", out_dir)

    print("All figures created.")


if __name__ == "__main__":
    main()
