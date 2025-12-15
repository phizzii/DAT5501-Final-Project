from datasets import load_dataset

dataset_stream = load_dataset("parquet", data_files="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw_review_All_Beauty/*.parquet", split="train", streaming=True)

