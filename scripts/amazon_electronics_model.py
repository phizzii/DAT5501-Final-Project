import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime
from datasets import load_from_disk

# modules for training model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# converts numeric rating into a binary target
def label_negative_experience(rating):
    if rating <= 2:
        return 1
    elif rating >= 4:
        return 0
    else:
        return np.nan # dropping neutral reviews
    
# takes one review text string and returns row of text features
# text features designed the capture the negative signal more than the rating does
# text features used: sentiment, length, punctuation intensity and caps intensity
def extract_text_features(text):
    # making sure text is a usable string
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # remove whitespace
    text = text.strip()
    
    # if string is empty after removing whitespace, return default values
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

    # textblob calculates sentiment and sentence splitting (with nltk)
    blob = TextBlob(text)

    # token with whitespace to get the word count
    words = text.split()

    # sentence objects from textblob
    sentences = blob.sentences

    # average sentence length in words
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
        "capital_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1) # calculate ratio of upper case by dividing number in caps by total text
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
    
    # drop neutral reviews
    dataframe = dataframe.dropna(subset=["negative_experience"])

    # convert to int labels
    dataframe["negative_experience"] = dataframe["negative_experience"].astype(int)

    # convert timestamp to see how old the review is (in days)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms")

    # in days relative to current time
    dataframe["review_age_days"] = (datetime.now() - dataframe["timestamp"]).dt.days

    # extract the text features
    # reset index so alignment is clean
    dataframe = dataframe.reset_index(drop=True)

    # apply feature extractor to each review
    text_features = dataframe["text"].apply(extract_text_features)

    # another reset index
    text_features = text_features.reset_index(drop=True)

    # merge into main dataframe
    dataframe = pd.concat([dataframe, text_features], axis=1)

    # final clean up,,, i hope (remove raw text and raw timestamp as i have info i need)
    dataframe = dataframe.drop(columns=["text", "timestamp"])

    # data checks for my own sanity
    print("rows:", len(dataframe))
    print("missing per column:\n", dataframe.isna().sum().sort_values(ascending=False).head(10))
    print("target balance\n", dataframe["negative_experience"].value_counts(normalize=True))

    # model training!
    # feature columns for ml
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

    # numeric subset except verified purchase (basically everything else)
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

    # x = features, y = target
    x = dataframe[feature_cols]
    y = dataframe["negative_experience"].astype(int)

    # ohe boolean column (verified purchase), standardscaler all numeric columns so i can make coefficients comparable and ignore any columns not explicitly transformed
    preprocessor = ColumnTransformer(
        transformers=[("verified_ohe", OneHotEncoder(drop="if_binary"), ["verified_purchase"]), ("num_scaler", StandardScaler(), numeric_features)], remainder="drop")

    # model pipeline: preprocess. logistic regression
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", # for class imbalance
            max_iter=2000, # higher iterations for convergence
            random_state=33 # random state, 33 is not technically important, my favourite number
        ))
    ])

    # train/test split with stratify to keep class proportions consistent
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=33)

    # fit pipeline
    model.fit(x_train, y_train)

    # predict labels on test set
    y_pred = model.predict(x_test)
    
    # feature names change
    final_feature_names = model.named_steps["preprocess"].get_feature_names_out().tolist()

    # put them into sorted series so i can print it
    feature_importance = pd.Series(
        model.named_steps["classifier"].coef_[0],
        index=final_feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
    
    print(feature_importance)

    print(classification_report(y_test, y_pred, zero_division=0))

    # save outputs
    output_path = "datasets/processed/amazon_electronics_features.csv"
    dataframe.to_csv(output_path, index=False)

    print("processed feature dataset save to: " + output_path)

    feature_names = model.named_steps["preprocess"].get_feature_names_out().tolist()
    
    # logistic regression coefficients (one per final feature)
    coefs = model.named_steps["classifier"].coef_[0]

    # save coefficients to csv for later use
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    }).sort_values("coef")

    coef_df.to_csv("coefs/amazon_electronics_coefs.csv", index=False)

if __name__ == "__main__":
    main()