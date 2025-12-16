from datasets import load_dataset
import random

dataset = load_dataset("Yelp/yelp_review_full", split="train")

print(dataset.features)

random.seed(42)

sample_size = 25_000
dataset_sampled = dataset.shuffle(seed=42).select(range(sample_size))

print(dataset_sampled)

# SAVING THE DATASET TO MY REPO SO I DON'T NEED TO RUN THIS SCRIPT AGAIN
dataset_sampled.save_to_disk("datasets/yelp_reviews_25k")
