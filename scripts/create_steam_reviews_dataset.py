# necessary modules imported for script
import os
import pandas as pd
import random
from datasets import Dataset
from langdetect import detect, DetectorFactory, LangDetectException

# setting langdect seed as apparently it can behave a little weird
DetectorFactory.seed = 33 # fav number, not technically important

# prevents script crashing when language detection fails and marks them as unknown
def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown" # marking up reviews which can't be detected

# path to the unzipped steam review files in raw data folder (which is in gitignore)
steam_reviews_dir = "raw_data/Game Reviews"

# sampling seed for reproducibility, 33 is my favourite number and not technically important
random.seed(33)

# sample size, bigger than amazon as there are more amazon datasets than steam
target_size = 25_000

# initialise sampled rows list
sampled_rows = []

# function lists all csv files in directory as when i downloaded the steam dataset from online it had a NUMBER of different files that needed collating
files = [
    f for f in os.listdir(steam_reviews_dir)
    if f.endswith(".csv")
]

# shuffles file order to not always sample the same games first
random.shuffle(files)

# function to loop over files, if bigger than the list size then stop collecting them
for file in files:
    if len(sampled_rows) >= target_size:
        break

    # creates path
    file_path = os.path.join(steam_reviews_dir, file)
    
    # reads csv into dataframe
    dataframe = pd.read_csv(file_path)

    # drops rows if the review text is missing
    dataframe = dataframe.dropna(subset=["review"])

    # converts the reviews to strings if they aren't already
    dataframe["review"] = dataframe["review"].astype(str)
    
    # removing whitespace and empty things since it complains if i don't
    dataframe = dataframe[dataframe["review"].str.strip() != ""]

    # adds language column
    dataframe["lang"] = dataframe["review"].apply(safe_detect)

    # only keep reviews in english
    dataframe = dataframe[dataframe["lang"] == "en"]

    # makes sure the columns are always strings and not nan
    for col in ["user", "post_date", "review", "recommend", "early_access_review"]:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].fillna("").astype(str)

    # shuffles row order inside file, 33 is not technically important, just my favourite number
    dataframe = dataframe.sample(frac=1, random_state=33)

    # converts each row into a dict and then appends it to sampled rows
    for _, row in dataframe.iterrows():
        sampled_rows.append(row.to_dict())
        if len(sampled_rows) >= target_size:
            break

# builds Hugging Face dataset from list of dicts
steam_dataset = Dataset.from_list(sampled_rows)
print(steam_dataset)

# SAVING THE DATASET TO MY REPO SO I DON'T NEED TO RUN THIS SCRIPT AGAIN
steam_dataset.save_to_disk("datasets/steam_reviews_25k")