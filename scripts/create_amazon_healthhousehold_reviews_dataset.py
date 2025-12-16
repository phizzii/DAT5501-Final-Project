import random
from itertools import islice
from datasets import load_dataset
from datasets import Dataset

dataset_stream = load_dataset("json", data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Health_and_Household.jsonl", split="train", streaming=True)

first = next(iter(dataset_stream))
print(first.keys())

random.seed(42)

sample_size = 10_000

sampled = list(islice(dataset_stream.shuffle(buffer_size=100_000, seed=42), sample_size))

dataset = Dataset.from_list(sampled)
print(dataset)

# SAVING THE DATASET TO MY REPO SO I DON'T NEED TO RUN THIS SCRIPT AGAIN
dataset.save_to_disk("datasets/amazon_health_reviews_10k")