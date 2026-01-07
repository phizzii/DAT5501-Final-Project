# import necessary modules
import os
import pandas as pd
import matplotlib.pyplot as plt

def clean_feature_name(name: str) -> str:
    # remove sklearn pipeline prefixes
    prefixes = [
        "num_scaler__", "remainder__", "verified_ohe__", "early_access_ohe__", "cat_ohe__"
    ]
    for p in prefixes:
        name = name.replace(p, "")

    # make it readable, no underscores n stuff
    name = name.replace("_", " ").strip()

    # fix specific annoying one-hot label from Steam
    name = name.replace("early access review Early Access Review", "Early access review")
    name = name.replace("Early access review Early Access Review", "Early access review")

    # title-case
    return name[:1].upper() + name[1:]

# loading the coefficient datasets
def load_coefs(path: str, platform_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["platform"] = platform_name
    
    # apply format function for clean and readable labels
    df["feature_clean"] = df["feature"].apply(clean_feature_name)
    return df[["platform", "feature_clean", "coef"]]


def main():
    coef_files = {
        "Amazon (Beauty)": "coefs/amazon_beauty_coefs.csv",
        "Steam": "coefs/steam_coefs.csv",
        "Yelp": "coefs/yelp_coefs.csv",
    }

    # load and combine
    frames = []
    for platform, path in coef_files.items():
        if not os.path.exists(path):
            print(f"missing file: {path}")
            return
        frames.append(load_coefs(path, platform))

    # merge em all together for big dataframe and graph building
    all_df = pd.concat(frames, ignore_index=True)

    # choose top features globally (by absolute coefficient), so the figure is comparable
    top_n = 8
    top_features = (
        all_df.assign(abs_coef=all_df["coef"].abs())
             .groupby("feature_clean")["abs_coef"].max()
             .sort_values(ascending=False)
             .head(top_n)
             .index
    )

    plot_df = all_df[all_df["feature_clean"].isin(top_features)].copy()

    # pivot to wide format for plotting: rows=features, cols=platforms
    wide = plot_df.pivot_table(
        index="feature_clean",
        columns="platform",
        values="coef",
        aggfunc="first"
    ).fillna(0.0)

    # order features by max absolute coefficient (best readability)
    wide = wide.loc[wide.abs().max(axis=1).sort_values(ascending=True).index]

    # plot it!
    ax = wide.plot(kind="barh", figsize=(11, 7))
    ax.set_title("Key Figure: Top predictors of negative user experience (standardised LR coefficients)")
    ax.set_xlabel("Coefficient (standardised features)")
    ax.set_ylabel("")  # cleaner
    plt.tight_layout()

    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "key_figure_cross_platform_coefficients.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
