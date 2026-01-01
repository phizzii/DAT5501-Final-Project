from multiprocessing import Pipe
import random
import pandas as pd
import numpy as np
from regex import D
from sqlalchemy import asc
from textblob import TextBlob
from datetime import date, datetime
from datasets import Dataset, load_dataset, load_from_disk

# modules for training model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def label_negative_experience(recommend):
    if recommend == "Not Recommended" :
        return 1
    elif recommend == "Recommended":
        return 0
    else:
        return np.nan # dropping neutral reviews
    
def extract_text_features(text):
    # making sure text is a usable string
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.strip()
    if len(text) == 0:
        # return safe defaults
        return pd.Series({
        "sentiment_polarity": 0.0,
        "sentiment_subjectivity": 0.0,
        "review_length_words": 0,
        "review_length_chars": 0,
        "avg_sentence_length": 0.0,
        "exclamation_count": 0,
        "capital_ratio": 0.0
    })

    blob = TextBlob(text)
    words = text.split()
    sentences = blob.sentences

    if len(sentences) == 0:
        avg_sentence_length = 0.0
    else:
        avg_sentence_length = float(np.mean([len(s.words) for s in sentences]))

    return pd.Series({
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity,
        "review_length_words": len(words),
        "review_length_chars": len(text),
        "avg_sentence_length": np.mean([len(s.words) for s in sentences]),
        "exclamation_count": text.count("!"),
        "capital_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1)
    })


# writing as main function because then we can use circleci testing dashboard later
def main():
    # loading sampled steam dataset
    dataframe_arrow = load_from_disk("datasets/steam_reviews_25k")
    dataframe = dataframe_arrow.to_pandas()
    dataframe.to_csv("datasets/csvs/steam_reviews.csv", index=False)

    # keeping the main columns i need
    dataframe = dataframe[[
        "playtime",
        "helpfulness",
        "review",
        "post_date",
        "recommend",
        "early_access_review"
    ]]

    # dropping missing text
    dataframe = dataframe.dropna(subset=["review"])

    # labelling target
    dataframe["negative_experience"] = dataframe["recommend"].apply(label_negative_experience)
    dataframe = dataframe.dropna(subset=["negative_experience"])
    dataframe["negative_experience"] = dataframe["negative_experience"].astype(int)

    # convert post_date to see how old the review is
    dataframe["post_date"] = pd.to_datetime(dataframe["post_date"], errors="coerce")
    dataframe = dataframe.dropna(subset=["post_date"])
    dataframe["review_age_days"] = (pd.Timestamp.now() - dataframe["post_date"]).dt.days

    # extract the text features
    dataframe = dataframe.reset_index(drop=True)
    text_features = dataframe["review"].apply(extract_text_features)
    text_features = text_features.reset_index(drop=True)
    dataframe = pd.concat([dataframe, text_features], axis=1)

    dataframe = dataframe.drop(columns=["review", "post_date"])

    print("rows:", len(dataframe))
    print("missing per column:\n", dataframe.isna().sum().sort_values(ascending=False).head(10))
    print("target balance\n", dataframe["negative_experience"].value_counts(normalize=True))

    # model training!

    feature_cols = [
        "playtime",
        "helpfulness",
        "review_age_days",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "review_length_words",
        "review_length_chars",
        "avg_sentence_length",
        "exclamation_count",
        "capital_ratio"
    ]

    numeric_features = [
        "playtime",
        "helpfulness",
        "review_age_days",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "review_length_words",
        "review_length_chars",
        "avg_sentence_length",
        "exclamation_count",
        "capital_ratio"
    ]

    # cause this is a unique column, it either is or isn't early access
    categorical_features = [
        "early_access_review"
    ]

    x = dataframe[feature_cols + categorical_features]
    y = dataframe["negative_experience"]

    preprocessor = ColumnTransformer(
        transformers=[("early_access_ohe", OneHotEncoder(drop="if_binary"), ["early_access_review"]), ("num_scaler", StandardScaler(), numeric_features)], remainder="drop")

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=33
        ))
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=33)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    final_feature_names = model.named_steps["preprocess"].get_feature_names_out().tolist()

    feature_importance = pd.Series(
        model.named_steps["classifier"].coef_[0],
        index=final_feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
    
    print(feature_importance)

    print(classification_report(y_test, y_pred, zero_division=0))

main()