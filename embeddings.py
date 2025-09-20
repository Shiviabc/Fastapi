import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os





def create_and_save_embeddings():
    CSV_PATH = r"data/New folder/Engineering.csv"
    EMBEDDINGS_PATH = r"data/embed.npy"
    JSON_DATA_PATH = r"data/college_data.json"

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    print("Detected columns:", df.columns.tolist())

    # Ensure required columns exist
    required_cols = ["institute_short", "program_name", "category", "closing_rank"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Build description used for embeddings
    df["full_description"] = (
        df["institute_short"].astype(str)
        + " offers " + df["program_name"].astype(str)
        + " (" + df["category"].astype(str) + ") "
        + " with closing rank " + df["closing_rank"].astype(str)
    )

    print("Loading embedding model (this may take a moment)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings for the dataset...")
    embeddings = model.encode(df["full_description"].tolist(), show_progress_bar=True)

    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)

    # Save full structured data including closing_rank
    college_data = df.to_dict(orient="records")
    with open(JSON_DATA_PATH, "w") as f:
        json.dump(college_data, f, indent=2)

    print(f"Saved {len(college_data)} rows with embeddings + metadata to {JSON_DATA_PATH}")


if __name__ == "__main__":
    create_and_save_embeddings()
