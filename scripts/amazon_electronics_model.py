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

def label_negative_experience(rating):
    if rating <= 2:
        return 1
    elif rating >= 4:
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
    # loading sampled amazon electronics dataset
    dataframe_arrow = load_from_disk("datasets/amazon_electronic_reviews_10k")
    dataframe = dataframe_arrow.to_pandas()
    dataframe.to_csv("datasets/csvs/amazon_electronics.csv", index=False)

    # keeping the main columns i need
    dataframe = dataframe[[
        "rating",
        "text",
        "helpful_vote",
        "verified_purchase",
        "timestamp"
    ]]

    # dropping missing text
    dataframe = dataframe.dropna(subset=["text"])

    # labelling target
    dataframe["negative_experience"] = dataframe["rating"].apply(label_negative_experience)
    dataframe = dataframe.dropna(subset=["negative_experience"])
    dataframe["negative_experience"] = dataframe["negative_experience"].astype(int)

    # convert timestamp to see how old the review is
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
    dataframe["review_age_days"] = (datetime.now() - dataframe["timestamp"]).dt.days

    # extract the text features
    dataframe = dataframe.reset_index(drop=True)
    text_features = dataframe["text"].apply(extract_text_features)
    text_features = text_features.reset_index(drop=True)
    dataframe = pd.concat([dataframe, text_features], axis=1)

    # final clean up,,, i hope
    dataframe = dataframe.drop(columns=["text", "timestamp"])

    print("rows:", len(dataframe))
    print("missing per column:\n", dataframe.isna().sum().sort_values(ascending=False).head(10))
    print("target balance\n", dataframe["negative_experience"].value_counts(normalize=True))

    # model training!

    feature_cols = [
        "helpful_vote",
        "verified_purchase",
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
        "helpful_vote",
        "review_age_days",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "review_length_words",
        "review_length_chars",
        "avg_sentence_length",
        "exclamation_count",
        "capital_ratio"
    ]

    x = dataframe[feature_cols]
    y = dataframe["negative_experience"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[("verified_ohe", OneHotEncoder(drop="if_binary"), ["verified_purchase"]), ("num_scaler", StandardScaler(), numeric_features)], remainder="drop")

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

    output_path = "datasets/processed/amazon_electronics_features.csv"
    dataframe.to_csv(output_path, index=False)

    print("processed feature dataset save to: " + output_path)

    feature_names = model.named_steps["preprocess"].get_feature_names_out().tolist()
    
    coefs = model.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    }).sort_values("coef")

    coef_df.to_csv("coefs/amazon_electronics_coefs.csv", index=False)

main()