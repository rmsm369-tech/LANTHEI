import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import random
import os

print("Starting the dataset processing script...")

TRAIN_FILE = "imdb_train.parquet"
VALIDATION_FILE = "imdb_validation.parquet"
TEST_FILE = "imdb_test.parquet"

BATCH_SIZE = 1000

SCHEMA = pa.schema([
    pa.field('text', pa.string()),
    pa.field('label', pa.int64())
])

print("\n1. Loading IMDb dataset in streaming mode...")
imdb_train_stream = datasets.load_dataset("imdb", split="train", streaming=True)
imdb_test_stream = datasets.load_dataset("imdb", split="test", streaming=True)

full_dataset_stream = datasets.interleave_datasets([imdb_train_stream, imdb_test_stream])
print("Dataset loaded and prepared for streaming. Starting to process examples...")

split_data = {
    "train": {"buffer": [], "row_count": 0, "positive_count": 0, "negative_count": 0, "writer": None, "file_path": TRAIN_FILE},
    "validation": {"buffer": [], "row_count": 0, "positive_count": 0, "negative_count": 0, "writer": None, "file_path": VALIDATION_FILE},
    "test": {"buffer": [], "row_count": 0, "positive_count": 0, "negative_count": 0, "writer": None, "file_path": TEST_FILE},
}

print("\n2 & 3. Splitting dataset and saving each split as a Parquet file locally...")
for i, example in enumerate(full_dataset_stream):
    rand_num = random.random()
    current_split_name = None

    if rand_num < 0.8:
        current_split_name = "train"
    elif rand_num < 0.9:
        current_split_name = "validation"
    else:
        current_split_name = "test"

    split_data[current_split_name]["row_count"] += 1
    if example["label"] == 1:
        split_data[current_split_name]["positive_count"] += 1
    else:
        split_data[current_split_name]["negative_count"] += 1

    split_data[current_split_name]["buffer"].append(example)

    if len(split_data[current_split_name]["buffer"]) >= BATCH_SIZE:
        data_to_write_df = pd.DataFrame(split_data[current_split_name]["buffer"])
        table_to_write = pa.Table.from_pandas(data_to_write_df, schema=SCHEMA)

        if split_data[current_split_name]["writer"] is None:
            split_data[current_split_name]["writer"] = pq.ParquetWriter(
                split_data[current_split_name]["file_path"],
                SCHEMA
            )
        split_data[current_split_name]["writer"].write_table(table_to_write)
        split_data[current_split_name]["buffer"] = []

    if (i + 1) % 5000 == 0:
        print(f"  Processed {i + 1} examples...")

print("\nFinished processing all examples. Writing any remaining buffered data...")

for split_name, data in split_data.items():
    if len(data["buffer"]) > 0:
        data_to_write_df = pd.DataFrame(data["buffer"])
        table_to_write = pa.Table.from_pandas(data_to_write_df, schema=SCHEMA)
        if data["writer"] is None:
            data["writer"] = pq.ParquetWriter(data["file_path"], SCHEMA)
        data["writer"].write_table(table_to_write)
    if data["writer"] is not None:
        data["writer"].close()
        print(f"  Closed ParquetWriter for {split_name} split.")
    elif os.path.exists(data["file_path"]):
        if os.path.getsize(data["file_path"]) == 0:
            os.remove(data["file_path"])
            print(f"  Removed empty Parquet file: {data['file_path']}")

print("\n--- 4. Dataset Split Summaries ---")
for split_name, data in split_data.items():
    print(f"\n{split_name.capitalize()} Split:")
    print(f"  Total Rows: {data['row_count']}")
    print(f"  Label Distribution:")
    positive_percentage = (data['positive_count'] / data['row_count']) * 100 if data['row_count'] > 0 else 0
    negative_percentage = (data['negative_count'] / data['row_count']) * 100 if data['row_count'] > 0 else 0
    print(f"    Positive (label 1): {data['positive_count']} ({positive_percentage:.2f}%)")
    print(f"    Negative (label 0): {data['negative_count']} ({negative_percentage:.2f}%)")

print("\nScript finished successfully!")