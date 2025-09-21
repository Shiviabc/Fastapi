import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os


def create_and_save_embeddings():
    CSV_PATH = os.path.join("data", "New folder", "combined_college_data.csv")
    EMBEDDINGS_PATH = os.path.join("data", "embed.npy")
    JSON_DATA_PATH = os.path.join("data", "college_data.json")

    print(f"Attempting to load dataset from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"FATAL ERROR: The file was not found at the path '{CSV_PATH}'.")
        return

    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    print("Detected columns:", df.columns.tolist())

    # --- FIX: Add 'city' and 'state' to the required columns ---
    required_cols = ["institute_short", "stream", "category", "closing_rank", "quota", "city", "state"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CRITICAL ERROR: The dataset is missing the required column: '{col}'")

    # --- FIX: Create a richer description including city and state ---
    df["full_description"] = (
            df["institute_short"].astype(str)
            + " offers " + df["stream"].astype(str)
            + " in " + df["city"].astype(str)
            + ", " + df["state"].astype(str)
            + " under the " + df["quota"].astype(str)
            + " quota for the " + df["category"].astype(str) + " category"
    )

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Generating embeddings...")
    embeddings = model.encode(df["full_description"].tolist(), show_progress_bar=True)

    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    np.save(EMBEDDINGS_PATH, embeddings)

    college_data = df.to_dict(orient="records")
    with open(JSON_DATA_PATH, "w") as f:
        json.dump(college_data, f, indent=2)

    print(f"Saved {len(college_data)} rows to {JSON_DATA_PATH}")


if __name__ == "__main__":
    create_and_save_embeddings()