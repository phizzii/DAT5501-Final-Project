# graphs (3 or 4 maybe) to support my hypotheses by giving evidence that goes with them
# for h1: bar chart of top coefficients from logistic regression per dataset
# for h2: boxplots of sentiment polarity, review length in words, exclamation mark count across amazon vs steam vs yelp with negative experience set to one
# for h3: example early warning proxy

import os
import pandas as pd
import matplotlib.pyplot as plt

# function to make names clear cause everything comes with numscaler from standard scaler
def clean_feature_name(raw_name: str) -> str:
    name = raw_name

    # remove pipeline prefixes
    for prefix in ["num_scaler__", "remainder__", "verified_ohe__", "early_access_ohe__", "cat_ohe__"]:
        name = name.replace(prefix, "")

    # nicer formatting
    name = name.replace("_", " ").strip()

    # capitalise a bit
    name = name[:1].upper() + name[1:]

    # special cases
    name = name.replace("Sentiment polarity", "Sentiment polarity")
    name = name.replace("Sentiment subjectivity", "Sentiment subjectivity")
    name = name.replace("Review length words", "Review length (words)")
    name = name.replace("Review length chars", "Review length (chars)")
    name = name.replace("Avg sentence length", "Avg sentence length")
    name = name.replace("Exclamation count", "Exclamation count")
    name = name.replace("Capital ratio", "Capital ratio")

    # handle one-hot outputs like "verified purchase True"
    name = name.replace("Verified purchase true", "Verified purchase: True")
    name = name.replace("Verified purchase false", "Verified purchase: False")
    name = name.replace("Early access review Early access review", "Early Access Review")

    return name


def dataset_display_name(dataset_key: str) -> str:
    # plot titles and such
    mapping = {
        "amazon_health": "Amazon (Health)",
        "amazon_beauty": "Amazon (Beauty)",
        "amazon_electronics": "Amazon (Electronics)",
        "amazon_clothing": "Amazon (Clothing)",
        "steam": "Steam",
        "yelp": "Yelp"
    }
    return mapping.get(dataset_key, dataset_key)

# fig 1 (h1): top coefficients
def plot_top_coefficients(coef_csv_path: str, dataset_key: str, out_dir: str, top_n: int = 8):
    df = pd.read_csv(coef_csv_path)

    # take largest by absolute value
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).head(top_n)

    # clean labels
    df["feature_clean"] = df["feature"].apply(clean_feature_name)

    # plot
    plt.figure(figsize=(10, 6))
    plt.barh(df["feature_clean"][::-1], df["coef"][::-1])
    plt.title(f"{dataset_display_name(dataset_key)}: Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient (standardised feature space)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"fig1_{dataset_key}_top_coefficients.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

# fig 2 (h2): cross-platform distributions (negative only)
def trim_series(s: pd.Series, low_q=0.01, high_q=0.99) -> pd.Series:
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s[(s >= lo) & (s <= hi)]

def violin_by_platform(df: pd.DataFrame, value_col: str, title: str, out_path: str):
    # df must have columns: platform, value_col
    platforms = sorted(df["platform"].unique())

    data = []
    for p in platforms:
        s = df.loc[df["platform"] == p, value_col].dropna()
        s = trim_series(s, 0.01, 0.99)  # trim extremes
        data.append(s.values)

    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=True, showmedians=True)

    plt.xticks(range(1, len(platforms) + 1), platforms, rotation=0)
    plt.title(title + " (trimmed 1st - 99th percentile)")
    plt.ylabel(value_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# fig 3 (h3): text signal vs rating level / recommend group
def plot_text_signal_vs_rating(feature_csv_path: str, dataset_key: str, out_dir: str):
    df = pd.read_csv(feature_csv_path)

    # choose x-axis grouping depending on dataset
    if dataset_key.startswith("amazon"):
        if "rating" not in df.columns:
            print(f"Skipping {dataset_key} (no rating column).")
            return
        x_col = "rating"
        # bucket ratings: 1,2,3,4,5 (already like that but maybe float)
        df["rating_bucket"] = df[x_col].round().astype(int)
        group_col = "rating_bucket"
        group_order = sorted(df[group_col].dropna().unique())

        x_label = "Rating"
        title = f"{dataset_display_name(dataset_key)}: Sentiment vs Rating"

    elif dataset_key == "yelp":
        if "label" not in df.columns:
            print("Skipping Yelp (no label column).")
            return
        # yelp label 0-4; interpret as star = label+1 for readability
        df["stars"] = df["label"].astype(int) + 1
        group_col = "stars"
        group_order = sorted(df[group_col].dropna().unique())

        x_label = "Star rating"
        title = "Yelp: Sentiment vs Star Rating"

    elif dataset_key == "steam":
        # if theres recommend text in the saved feature CSV then use it
        # if theres not, fallback to negative_experience (0/1).
        if "recommend" in df.columns:
            group_col = "recommend"
            group_order = ["Recommended", "Not Recommended"]
            x_label = "Recommendation"
            title = "Steam: Sentiment vs Recommendation"
        else:
            group_col = "negative_experience"
            group_order = [0, 1]
            x_label = "Negative experience (0/1)"
            title = "Steam: Sentiment vs Negative Experience"
    else:
        print(f"Unknown dataset key: {dataset_key}")
        return

    if "sentiment_polarity" not in df.columns:
        print(f"Skipping {dataset_key} (no sentiment_polarity column).")
        return

    # make boxplot data in order
    data = []
    labels = []
    for g in group_order:
        subset = df[df[group_col] == g]["sentiment_polarity"].dropna()
        if len(subset) == 0:
            continue
        data.append(subset)
        labels.append(str(g))

    if not data:
        print(f"Skipping {dataset_key} (no data after grouping).")
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Sentiment polarity")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"fig3_{dataset_key}_sentiment_vs_group.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

# my lovely main function
def main():
    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)

    coef_paths = {
        "amazon_health": "coefs/amazon_health_coefs.csv",
        "amazon_beauty": "coefs/amazon_beauty_coefs.csv",
        "amazon_electronics": "coefs/amazon_electronics_coefs.csv",
        "amazon_clothing": "coefs/amazon_clothing_coefs.csv",
        "steam": "coefs/steam_coefs.csv",
        "yelp": "coefs/yelp_coefs.csv",
    }

    feature_paths = {
        "amazon_health": "datasets/processed/amazon_health_features.csv",
        "amazon_beauty": "datasets/processed/amazon_beauty_features.csv",
        "amazon_electronics": "datasets/processed/amazon_electronics_features.csv",
        "amazon_clothing": "datasets/processed/amazon_clothing_features.csv",
        "steam": "datasets/processed/steam_features.csv",
        "yelp": "datasets/processed/yelp_features.csv",
    }

    # fig 1: top coefficients per dataset
    for key, path in coef_paths.items():
        if os.path.exists(path):
            plot_top_coefficients(path, key, out_dir, top_n=8)
        else:
            print(f"Missing coef file for {key}: {path}")

    # fig 2: cross-platform distributions (picked only one amazon dataset since theres many of em, beauty'll do)
    fig2_map = {
        "amazon_beauty": feature_paths["amazon_beauty"],
        "steam": feature_paths["steam"],
        "yelp": feature_paths["yelp"],
    }
    plot_top_coefficients(fig2_map, out_dir)

    # fig 3: sentiment vs rating/recommend (one per platform type, again, one for amazon since theres many and beauty will be fine for this tbh)
    plot_text_signal_vs_rating(feature_paths["amazon_beauty"], "amazon_beauty", out_dir)
    plot_text_signal_vs_rating(feature_paths["yelp"], "yelp", out_dir)
    plot_text_signal_vs_rating(feature_paths["steam"], "steam", out_dir)

    print(f"Done. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
