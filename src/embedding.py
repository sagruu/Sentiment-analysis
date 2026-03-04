import numpy as np
import os
from tqdm import tqdm

def generate_and_save_embeddings_from_splits(splits, embedding_model, output_path, prefix):
    """
    Generiert und speichert Embeddings für bereits vorhandene DataFrame-Splits.

    Parameter:
    - splits: Dictionary {size: DataFrame} mit 'text'-Spalte
    - embedding_model: ein SentenceTransformer-Modell
    - output_path: Zielordner für die .npy-Dateien
    - prefix: Präfix für die Dateinamen
    """
    os.makedirs(output_path, exist_ok=True)

    for size, subset_df in tqdm(splits.items(), desc="Embedding Splits"):
        texts = subset_df["text"].tolist()
        embeddings = embedding_model.encode(texts, show_progress_bar=True)

        file_path = os.path.join(output_path, f"{prefix}_embeddings_size_{size}.npy")
        np.save(file_path, embeddings)
        print(f"Saved embeddings for size {size}: {file_path}")



