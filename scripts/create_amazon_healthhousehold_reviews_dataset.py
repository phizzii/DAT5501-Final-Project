# necessary modules imported for script
import random
from itertools import islice
from datasets import load_dataset
from datasets import Dataset

# loading the beauty and personal care data from the amazon reviews of 2023 from Hugging Face (which is where it is hosted online)
# use of streaming to make sure it is not fully downloaded as it is a large set of data and read each review line by line
dataset_stream = load_dataset("json", data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/Health_and_Household.jsonl", split="train", streaming=True)

# use of an iterator converted from stream (so i can access the reviews in the collection)
# pulls one review and prints the column name, confirm available fields
first = next(iter(dataset_stream))
print(first.keys())

# random seed for reproducibility, 33 is not technically important, its just my favourite number
random.seed(33)

# number of samples i want (since there is a lot of amazon reviews, i chose to have only 10,000 from each)
sample_size = 10_000

# shuffles streaming dataset, loads 100k and shuffles in the buffer for approx sampling
sampled = list(islice(dataset_stream.shuffle(buffer_size=100_000, seed=42), sample_size))

# convert to Hugging Face dataset so i can save to disk and don't need to reload and i can have a reusable dataset
dataset = Dataset.from_list(sampled)
print(dataset)

# SAVING THE DATASET TO MY REPO SO I DON'T NEED TO RUN THIS SCRIPT AGAIN
dataset.save_to_disk("datasets/amazon_health_reviews_10k")