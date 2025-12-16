import os
import pandas as pd
import random
from datasets import Dataset

# path to the unzipped steam review files in raw data folder (which is in gitignore)
steam_reviews_dir = "raw_data/Game Reviews"

random.seed(42)
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

    dataframe = dataframe.sample(frac=1, random_state=42)

    for _, row in dataframe.iterrows():
        sampled_rows.append(row.to_dict())
        if len(sampled_rows) >= target_size:
            break

steam_dataset = Dataset.from_list(sampled_rows)
print(steam_dataset)
