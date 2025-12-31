# my model is a review-level analytical framework to detect early signs of negative user experience and platform-specific bias, by analysing linguistic and sentiment features in user-generated reviews, enabling proactive reputation and product management
# hypothetically, a company would be able to monitor INCOMING reviews automatically, detect emerging dissatisfaction patterns early, understand how negativity is expressed differently per platform (diff user bases) and act before average ratings or sales metrics visibly decline

# le plan
# discrete outcome (negative experience) defined as 'rating' being: <= 2 (negative), >= 4 (positive) and == 3 (neutral)

# feature groups

# putting raw text isn't really plausible so there's going to be sentiment features which are interpretability focused like the proportion of negative words used (business meaning being how emotionally negative is the review?)
# linguistic intensity features, like review length (word count), maybe number of symbols used (like exclamation marks), how much capitalisation is there?, use of intensifiers like 'very', 'extremely' or 'worst' (business meaning being how strongly is the user expressing negativity?)
# structural features, like number of stars or how many people found the review useful? (not sure about that one though), platform indicator (business meaning being where is the dissatisfaction coming from)
# optional lexical features, like frequency of complaint related terms like refunds, broken or crash (but only to help not dominate)

# model options (all used not just one): decision tree (because feature importance for business insight), random forest (for robustness and performance comparison), logistic regression (as a baseline checker and interpretability)

# evaluation metrics: the aim is not for perfect accuracy so i'm going to use precision because sometimes models can give false positives!! (which would be costly for a business), recall because sometimes negative reviews may be missed which happens to also be costly to the business (you can see where this is going)

# how i'm mapping these to my hypotheses:
# H1
# analysis: label reviews using star ratings (maybe maybe not), extract sentiment & linguistic features, train classifier (decision tree/logic), evaluate performance (precision & recall), inspect feature importance
# evidence presented: classification report, feature importance plot, discussion of which features dominate

# H2
# analysis: filter only negative reviews, group by platform, compare feature distributions (length, sentiment intensity, linguistic markers), train platform specific models or include platform as a feature (not sure yet)
# evidence presented: boxplot/density plots per platform, feature importance differences, model performance differences

# H3
# analysis: order reviews temporally, track the following; sentiment trend, linguistic intensity trend, average rating trend
# evidence presented: time series plots, qualitative interpretation

# negative_experience = 1 if rating <= 2
# negative_experience = 0 if rating >= 4
# rating = 3 > excluded as neutral reviews are ambiguous

import pandas as pd
import numpy as np
from textblob import TextBlob, download_corpora
from datetime import date, datetime
from datasets import Dataset, load_dataset, load_from_disk

def label_negative_experience(rating):
    if rating <= 2:
        return 1
    elif rating >= 4:
        return 0
    else:
        return np.nan # dropping neutral reviews
    
def extract_text_features(text):
    blob = TextBlob(text)
    words = text.split()
    sentences = blob.sentences

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
    # loading sampled amazon beauty and personal care dataset
    dataframe_arrow = load_from_disk("datasets/amazon_beauty_reviews_10k")
    dataframe = dataframe_arrow.to_pandas()
    dataframe.to_csv("datasets/csvs/amazon_beauty.csv", index=False)

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

    # convert timestamp to see how old the review is
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], unit="ms")
    dataframe["review_age_days"] = (datetime.now() - dataframe["timestamp"]).dt.days

    # extract the text features
    text_features = dataframe["text"].apply(extract_text_features)
    dataframe = pd.concat([dataframe.reset_index(drop=True), text_features], axis=1)

    # final clean up,,, i hope
    dataframe = dataframe.drop(columns=["text", "timestamp"])

    print(dataframe.head())
    print(dataframe.info())

main()