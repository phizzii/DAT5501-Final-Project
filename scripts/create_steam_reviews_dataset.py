import os
import pandas as pd
import random
from datasets import Dataset
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 33 # fav number, not technically important

def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown" # marking up reviews which can't be detected

# path to the unzipped steam review files in raw data folder (which is in gitignore)
steam_reviews_dir = "raw_data/Game Reviews"

random.seed(33) # fav number, not technically important
target_size = 25_000

sampled_rows = []

files = [
    f for f in os.listdir(steam_reviews_dir)
    if f.endswith(".csv")
]

random.shuffle(files)

for file in files:
    if len(sampled_rows) >= target_size:
        break

    file_path = os.path.join(steam_reviews_dir, file)
    
    dataframe = pd.read_csv(file_path)

    dataframe = dataframe.dropna(subset=["review"])

    dataframe["review"] = dataframe["review"].astype(str)
    
    dataframe = dataframe[dataframe["review"].str.strip() != ""] # removing whitespace and empty things since it complains if i don't

    dataframe["lang"] = dataframe["review"].apply(safe_detect)

    dataframe = dataframe[dataframe["lang"] == "en"]

    for col in ["user", "post_date", "review", "recommend", "early_access_review"]:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].fillna("").astype(str)

    dataframe = dataframe.sample(frac=1, random_state=42)

    for _, row in dataframe.iterrows():
        sampled_rows.append(row.to_dict())
        if len(sampled_rows) >= target_size:
            break

steam_dataset = Dataset.from_list(sampled_rows)
print(steam_dataset)

# SAVING THE DATASET TO MY REPO SO I DON'T NEED TO RUN THIS SCRIPT AGAIN
steam_dataset.save_to_disk("datasets/steam_reviews_25k")