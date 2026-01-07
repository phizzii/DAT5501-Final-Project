import pandas as pd
import numpy as np
from textblob import TextBlob
from datasets import load_from_disk

# modules for training model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# convert label into (numeric) into binary target
def label_negative_experience(label):
    if label <= 2:
        return 1
    elif label >= 4:
        return 0
    else:
        return np.nan # dropping neutral reviews
    
# take single review text string and returns row of text features
# text features designed to capture negative signals beyond the rating: sentiment, length, punctuation intensity and caps intensity
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

    # textblob computes sentiment and sentence splitting (with nltk)
    blob = TextBlob(text)
    
    # whitespace as token for word count
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
        "avg_sentence_length": avg_sentence_length,
        "exclamation_count": text.count("!"),
        "capital_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1) # caps ratio calculated bu how may are upper case divided by total number of chars
    })


# writing as main function because then we can use circleci testing dashboard later
def main():
    # loading sampled yelp dataset
    dataframe_arrow = load_from_disk("datasets/yelp_reviews_25k")
    dataframe = dataframe_arrow.to_pandas()
    dataframe.to_csv("datasets/csvs/yelp_reviews.csv", index=False)

    # keeping the main columns i need
    dataframe = dataframe[[
        "label",
        "text",
    ]]

    # dropping missing text
    dataframe = dataframe.dropna(subset=["text"])

    # labelling target
    dataframe["negative_experience"] = dataframe["label"].apply(label_negative_experience)
    
    # drop rows where review text is missing
    dataframe = dataframe.dropna(subset=["negative_experience"])
    
    # convert to int labels
    dataframe["negative_experience"] = dataframe["negative_experience"].astype(int)

    # extract the text features
    # reset index so alignment is clean
    dataframe = dataframe.reset_index(drop=True)

    # apply text feature extractor to each review
    text_features = dataframe["text"].apply(extract_text_features)

    # reset index again
    text_features = text_features.reset_index(drop=True)

    # merge new columns into main dataframe
    dataframe = pd.concat([dataframe, text_features], axis=1)

    # remove raw text because i have the info i need already
    dataframe = dataframe.drop(columns=["text"])

    # checking for my sanity
    print("rows:", len(dataframe))
    print("missing per column:\n", dataframe.isna().sum().sort_values(ascending=False).head(10))
    print("target balance\n", dataframe["negative_experience"].value_counts(normalize=True))

    # model training!
    # feature columns for ml
    feature_cols = [
        "sentiment_polarity",
        "sentiment_subjectivity",
        "review_length_words",
        "review_length_chars",
        "avg_sentence_length",
        "exclamation_count",
        "capital_ratio"
    ]

    # numeric features except early access review
    numeric_features = [
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

    # preprocessing stuff, standardscaler on all numeric columns and drop any columns not transformed
    preprocessor = ColumnTransformer(
        transformers=[("num_scaler", StandardScaler(), numeric_features)], remainder="drop")

    # model pipeline: preprocess, logistic regression
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", # for class imbalance
            max_iter=2000, # higher iterations to help with convergence
            random_state=33 # random state, 33 is not technically important, just my favourite number
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

    # put features into a sorted series for printing
    feature_importance = pd.Series(
        model.named_steps["classifier"].coef_[0],
        index=final_feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
    
    print(feature_importance)

    print(classification_report(y_test, y_pred, zero_division=0))

    # save output as csv
    output_path = "datasets/processed/yelp_features.csv"
    dataframe.to_csv(output_path, index=False)

    print("processed feature dataset save to: " + output_path)

    feature_names = model.named_steps["preprocess"].get_feature_names_out().tolist()
    
    # logistic regression coefficients (one per final feature)
    coefs = model.named_steps["classifier"].coef_[0]

    # save coefficients to csv for use later
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    }).sort_values("coef")

    coef_df.to_csv("coefs/yelp_coefs.csv", index=False)

if __name__ == "__main__":
    main()