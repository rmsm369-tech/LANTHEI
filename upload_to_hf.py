from huggingface_hub import HfApi
import os

api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/lanthei"

# Create the dataset repo (safe to run even if it already exists)
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# Upload each split file
for filename in ["imdb_train.parquet", "imdb_validation.parquet", "imdb_test.parquet"]:
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {filename}")

# Add a simple description (dataset card)
readme = """---
license: other
---
# Lanthei

A practice dataset built from the public IMDb dataset, split into train/validation/test (80/10/10).
Created as a learning project using streaming, no full download required.
"""
with open("README.md", "w") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
)
print("Done! Check huggingface.co/datasets/" + repo_id)