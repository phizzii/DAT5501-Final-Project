# import necessary modules
from datasets import load_dataset
import random

# gets the yelp full review dataset, loads only training split
dataset = load_dataset("Yelp/yelp_review_full", split="train")

# prints features so i know what columns there are (not many...)
print(dataset.features)

# set random seed for reproducibility, 33 is my favourite number, not technically important
random.seed(33)

# set sample size, again bigger than the amazon ones since there are many of those and only one of this one
sample_size = 25_000

# randomly reorders dataset and takes first 25k rows after shuffling
dataset_sampled = dataset.shuffle(seed=33).select(range(sample_size))

print(dataset_sampled)

# SAVING THE DATASET TO MY REPO SO I DON'T NEED TO RUN THIS SCRIPT AGAIN
dataset_sampled.save_to_disk("datasets/yelp_reviews_25k")