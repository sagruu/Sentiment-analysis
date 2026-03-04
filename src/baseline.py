

from tqdm import tqdm
import os
import json
import numpy as np
import torch
import random
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def split_nested_splits(nested_splits, test_size=0.2, val_size=0.1):
    """
    Splittet die Daten in Trainings-, Validierungs- und Test-Sets für verschiedene Dataset-Größen.
    
    Args:
        nested_splits (dict): Dictionary mit Dataset-Größen als Schlüsseln und DataFrames mit 'text' und 'label'.
        test_size (float): Anteil der Daten, die für den Test-Split verwendet werden (Standard: 0.2).
        val_size (float): Anteil der Trainingsdaten, die für den Validierungs-Split verwendet werden (Standard: 0.1).
        
    Returns:
        dict: Dictionary mit den gesplitteten Daten, wobei die Schlüssel die Dataset-Größen sind
              und die Werte Dictionaries mit 'train', 'val' und 'test' DataFrames.
              
    Raises:
        ValueError: Wenn die Eingabedaten nicht den erwarteten Format entsprechen oder leer sind.
    """
    split_data = {}

    for size, df in nested_splits.items():
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Data for size {size} is not a pandas DataFrame")
        if not {"text", "label"}.issubset(df.columns):
            raise ValueError(f"DataFrame for size {size} missing 'text' or 'label' columns")
        if df.empty:
            raise ValueError(f"DataFrame for size {size} is empty")

        # Filter invalid rows and log dropped rows
        original_len = len(df)
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].str.strip() != ""]
        if len(df) < original_len:
            print(f"Warning: Dropped {original_len - len(df)} invalid rows for size {size}")

        # Train-test split
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=42
        )

        # Train-validation split
        df_train, df_val = train_test_split(
            df_train,
            test_size=val_size / (1 - test_size),
            stratify=df_train["label"],
            random_state=42
        )

        split_data[size] = {
            "train": df_train.reset_index(drop=True),
            "val": df_val.reset_index(drop=True),
            "test": df_test.reset_index(drop=True)
        }

    return split_data

def run_tfidf_cv(split_data, save_path, save_json=True):
    """
    Führt TF-IDF Cross-Validation durch und evaluiert die Ergebnisse.
    
    Args:
        split_data (dict): Dictionary mit Dataset-Größen als Schlüsseln und DataFrames mit 'train' und 'test'.
        save_path (str): Verzeichnis, in dem die Ergebnisse gespeichert werden sollen.
        save_json (bool): Ob die Ergebnisse in einer JSON-Datei gespeichert werden sollen (Standard: True).
        
    Returns:
        dict: Ergebnisse mit den Metriken für jedes Dataset.
    
    Raises:
        ValueError: Wenn split_data leer ist oder die erforderlichen Spalten fehlen.
    """
    if not split_data:
        raise ValueError("split_data dictionary is empty")

    os.makedirs(save_path, exist_ok=True)
    results = {}

    for size, splits in split_data.items():
        print(f"\nRunning TF-IDF CV for size {size}")

        # Validate splits
        for split in ["train", "test"]:
            if split not in splits:
                raise ValueError(f"Missing {split} split for size {size}")

        # Fit vectorizer and transform training data
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(splits["train"]["text"])
        y = splits["train"]["label"]

        # Cross-validation
        clf = LogisticRegression(max_iter=1000)
        accs = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        f1s = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")

        # Train final model and evaluate on test set
        clf.fit(X, y)
        X_test = vectorizer.transform(splits["test"]["text"])
        y_test = splits["test"]["label"]
        preds_test = clf.predict(X_test)

        results[size] = {
            "cv_accuracy": float(np.mean(accs)),
            "cv_f1": float(np.mean(f1s)),
            "test_accuracy": accuracy_score(y_test, preds_test),
            "test_f1": f1_score(y_test, preds_test, average="weighted"),
            "classification_report": classification_report(y_test, preds_test, output_dict=True)
        }

    if save_json:
        try:
            with open(os.path.join(save_path, "tfidf_results.json"), "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving TF-IDF results: {e}")

    return results

def run_feature_extraction(split_data, model_name, save_path, save_json=True, batch_size=16):
    """
    Führt Feature-Extraktion mit einem vortrainierten Modell durch und evaluiert die Ergebnisse mit Cross-Validation.
    
    Args:
        split_data (dict): Dictionary mit Dataset-Größen als Schlüsseln und DataFrames mit 'train' und 'test'.
        model_name (str): Name des vortrainierten Modells (z.B. 'bert-base-uncased').
        save_path (str): Verzeichnis, in dem die Ergebnisse gespeichert werden sollen.
        save_json (bool): Ob die Ergebnisse in einer JSON-Datei gespeichert werden sollen (Standard: True).
        batch_size (int): Batch-Größe für die Verarbeitung der Texte (Standard: 16).
        
    Returns:
        dict: Ergebnisse mit den Metriken für jedes Dataset.
        
    Raises:
        ValueError: Wenn split_data leer ist oder die Modell- oder Tokenizer-Ladung fehlschlägt.
    """
    if not split_data:
        raise ValueError("split_data dictionary is empty")

    os.makedirs(save_path, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model or tokenizer: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def get_embeddings(texts):
        """Extracts mean-pooled embeddings from texts in batches."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(emb)
            # Log truncation if any text exceeds max_length
            if any(len(tokenizer.encode(t, add_special_tokens=True)) > 512 for t in batch_texts):
                print(f"Warning: Some texts in batch truncated for size {size}")
        return np.vstack(embeddings)

    results = {}

    for size, splits in split_data.items():
        print(f"\nRunning Feature Extraction for size {size}")

        # Validate splits
        for split in ["train", "test"]:
            if split not in splits:
                raise ValueError(f"Missing {split} split for size {size}")

        # Extract embeddings
        X = get_embeddings(splits["train"]["text"].tolist())
        y = splits["train"]["label"].values

        # Cross-validation
        clf = LogisticRegression(max_iter=1000)
        accs = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        f1s = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")

        # Train final model and evaluate on test set
        clf.fit(X, y)
        X_test = get_embeddings(splits["test"]["text"].tolist())
        y_test = splits["test"]["label"].values
        preds_test = clf.predict(X_test)

        results[size] = {
            "cv_accuracy": float(np.mean(accs)),
            "cv_f1": float(np.mean(f1s)),
            "test_accuracy": accuracy_score(y_test, preds_test),
            "test_f1": f1_score(y_test, preds_test, average="weighted"),
            "classification_report": classification_report(y_test, preds_test, output_dict=True),
            "model_params": clf.get_params(),  # <-- Speichert die LR-Parameter
            "embedding_info": {
                "train_shape": X.shape,
                "test_shape": X_test.shape,
                "model_name": model_name
            }
        }


    if save_json:
        try:
            with open(os.path.join(save_path, "feature_extraction_results.json"), "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving feature extraction results: {e}")

    return results

def run_fine_tuning(split_data, model_name, save_path, num_labels, save_json=True,  seed: int = 42):
    """
    Führt Fine-Tuning eines vortrainierten Modells mit Hugging Face Transformers durch.
    
    Args:
        split_data (dict): Dictionary mit Dataset-Größen als Schlüsseln und DataFrames mit 'train', 'val' und 'test'.
        model_name (str): Name des vortrainierten Modells (z.B. 'bert-base-uncased').
        save_path (str): Verzeichnis, in dem die Ergebnisse gespeichert werden sollen.
        num_labels (int): Anzahl der Klassen für die Klassifikation.
        save_json (bool): Ob die Ergebnisse in einer JSON-Datei gespeichert werden sollen (Standard: True).
        seed (int): Zufalls-Seed für Reproduzierbarkeit (Standard: 42).
    
    Returns:
        dict: Ergebnisse mit den Metriken für jedes Dataset.
    
    Raises:
        ValueError: Wenn split_data leer ist oder die Anzahl der Labels nicht mit den Daten übereinstimmt.
    """
    if not split_data:
        raise ValueError("split_data dictionary is empty")

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # gemeinsame Tokenisierungsfunktion (Batch!)
    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    results = {}

    num_proc = min(8, os.cpu_count())  # begrenze auf 8 Prozesse, falls Cluster

    for size, splits in split_data.items():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fine-Tuning für Size {size}")

        # Konsistenz-Checks
        for split in ("train", "val", "test"):
            if split not in splits:
                raise ValueError(f"Missing {split} split for size {size}")
        if len(splits["train"]["label"].unique()) != num_labels:
            raise ValueError(
                f"num_labels stimmt nicht (erwartet {num_labels}, gefunden "
                f"{len(splits['train']['label'].unique())})"
            )

        # DataSets schneller vorbereiten
        try:
            train_ds = (
                Dataset.from_pandas(splits["train"])
                .map(encode, batched=True, remove_columns=["text"], num_proc=num_proc)
            )
            val_ds = (
                Dataset.from_pandas(splits["val"])
                .map(encode, batched=True, remove_columns=["text"], num_proc=num_proc)
            )
            test_ds = (
                Dataset.from_pandas(splits["test"])
                .map(encode, batched=True, remove_columns=["text"], num_proc=num_proc)
            )
        except Exception as e:
            print(f"Tokenisierungsfehler für Size {size}: {e}")
            continue

        for ds in (train_ds, val_ds, test_ds):
            ds.set_format(type="torch")

        #  Modell 
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # TrainingArguments (ohne Checkpoint-Overhead) 
        training_args = TrainingArguments(
            output_dir=os.path.join(save_path, f"model_{size}"),
            evaluation_strategy="no",      # kein Eval während des Trainings
            save_strategy="no",            # Checkpoints erst am Ende
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            learning_rate=2e-5,
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",           # reiner PyTorch-AdamW
            dataloader_num_workers=os.cpu_count(),
            seed=seed,
            report_to="none",
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
        )

        try:
            trainer.train()
        except Exception as e:
            print(f"Trainingsfehler für Size {size}: {e}")
            continue

        # Manuell ein einziges Modell speichern
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Evaluate on validation and test sets
        val_metrics = trainer.evaluate(val_ds)
        test_out = trainer.predict(test_ds)
        test_metrics = test_out.metrics

        results[size] = {
            "val": val_metrics,
            "test": test_metrics,
            "log_history": trainer.state.log_history
        }

    if save_json:
        try:
            with open(os.path.join(save_path, "fine_tuning_results.json"), "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error saving fine-tuning results: {e}")

    return results




# Globale Konstanten
EMB_DIR_ORIG = "../Sentimentanalyse/outputs/embeddings"
EMB_DIR_CENTROID = "../Sentimentanalyse/outputs/embeddings/centroid"
EMB_DIR_ZERO = "../Sentimentanalyse/outputs/embeddings/zero"
FEAT_JSON_PATH = "../Sentimentanalyse/results/feature_extraction/feature_extraction_results.json"


def get_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    emb_dir: str,
    key: str
) -> np.ndarray:
    """
    Load or generate embeddings for the given texts.

    Parameters
    ----------
    texts : List[str]
        List of texts for which embeddings are to be generated.
    model : SentenceTransformer
        Model used for generating embeddings.
    emb_dir : str
        Directory where embeddings are to be stored.
    key : str
        Key for naming the embedding file.

    Returns
    -------
    np.ndarray
        Generated or loaded embeddings.

    Raises
    ------
    OSError
        If there is an issue with file operations.
    """
    os.makedirs(emb_dir, exist_ok=True)
    path = os.path.join(emb_dir, f"embeddings_{key}.npy")
    
    try:
        if os.path.exists(path):
            return np.load(path)
    except OSError as e:
        raise OSError(f"Error loading embeddings from {path}: {e}")

    try:
        embeddings = model.encode(texts, show_progress_bar=False)
        np.save(path, embeddings)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings for key {key}: {e}")


def pick_lr_params(
    feat_json: Dict,
    size: Optional[str] = None,
    metric_key: str = "test_f1"
) -> Dict:
    """
    Select the best logistic regression parameters based on feature extraction results.

    Parameters
    ----------
    feat_json : Dict
        JSON data with feature extraction results.
    size : Optional[str], optional
        Size of the split for which parameters are selected. If None, the size with the best metric is chosen.
    metric_key : str, optional
        Metric key used to select the parameters, by default "test_f1".

    Returns
    -------
    Dict
        Best logistic regression parameters for the specified size or globally best parameters.

    Raises
    ------
    KeyError
        If the specified size or metric_key is not found in feat_json.
    """
    if size is None:
        try:
            size = max(feat_json, key=lambda k: feat_json[k][metric_key])
        except KeyError as e:
            raise KeyError(f"Metric {metric_key} not found in feat_json")
    
    try:
        return feat_json[str(size)]["model_params"]
    except KeyError as e:
        raise KeyError(f"Size {size} not found in feat_json")


def train_with_pseudo_labels(
    split_df: Dict[str, pd.DataFrame],
    weak_labels_df: pd.DataFrame,
    model: SentenceTransformer,
    emb_dir_orig: str,
    emb_dir_weak: str,
    lr_params: Dict,
    weight_orig: float = 1.0,
    weight_weak: float = 1.0
) -> pd.DataFrame:
    """
    Train a model with pseudo-labels by combining original training data and weak labels.

    Parameters
    ----------
    split_df : Dict[str, pd.DataFrame]
        Dictionary with 'train', 'val', and 'test' DataFrames.
    weak_labels_df : pd.DataFrame
        DataFrame with weak labels used for pseudo-labeling.
    model : SentenceTransformer
        Model used for generating embeddings.
    emb_dir_orig : str
        Directory for original embeddings.
    emb_dir_weak : str
        Directory for weak embeddings.
    lr_params : Dict
        Parameters for logistic regression.
    weight_orig : float, optional
        Weight for original training data, by default 1.0.
    weight_weak : float, optional
        Weight for weak labels, by default 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions for the test split, including true labels,
        predicted labels, confidence, and label type.

    Raises
    ------
    ValueError
        If input DataFrames are empty or have incompatible shapes.
    """
    if not all(key in split_df for key in ["train", "test"]):
        raise ValueError("split_df must contain 'train' and 'test' DataFrames")
    if weak_labels_df.empty:
        raise ValueError("weak_labels_df cannot be empty")

    # Generate embeddings
    X_train_orig = get_embeddings(
        split_df["train"]["text"].tolist(), model, emb_dir_orig, f"{split_df['name']}_train"
    )
    y_train_orig = split_df["train"]["label"].values

    X_pseudo = get_embeddings(
        weak_labels_df["text"].tolist(), model, emb_dir_weak, f"{split_df['name']}_weak"
    )
    y_pseudo = weak_labels_df["weak_label"].values

    # Combine original and pseudo-labeled data with weights
    X_all = np.vstack([X_train_orig, X_pseudo])
    y_all = np.concatenate([y_train_orig, y_pseudo]).astype(int).ravel()
    sample_weights = np.concatenate([
        np.full(len(y_train_orig), weight_orig),
        np.full(len(y_pseudo), weight_weak)
    ])

    # Train logistic regression with sample weights
    clf = LogisticRegression(**lr_params)
    clf.fit(X_all, y_all, sample_weight=sample_weights)

    # Predict on test set
    X_test = get_embeddings(
        split_df["test"]["text"].tolist(), model, emb_dir_orig, f"{split_df['name']}_test"
    )
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test).max(axis=1)

    # Create output DataFrame
    df_out = split_df["test"].copy()
    df_out["true_label"] = df_out["label"]
    df_out["predicted_label"] = y_pred
    df_out["confidence"] = y_prob
    df_out["label_type"] = "hard"
    
    return df_out


def run_experiments_with_weights(
    split_data: Dict[str, Dict[str, pd.DataFrame]],
    labeled_splits_knn: Dict[str, pd.DataFrame],
    labeled_splits_zero: Dict[str, pd.DataFrame],
    model: SentenceTransformer,
    weight_orig: float = 1.0,
    weight_weak: float = 1.0,
    emb_dir_orig: str = EMB_DIR_ORIG,
    emb_dir_centroid: str = EMB_DIR_CENTROID,
    emb_dir_zero: str = EMB_DIR_ZERO,
    feat_json_path: str = FEAT_JSON_PATH
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Run experiments with pseudo-labels across all size splits with specified weights.

    Parameters
    ----------
    split_data : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary of size splits, each containing 'train', 'val', and 'test' DataFrames.
    labeled_splits_knn : Dict[str, pd.DataFrame]
        Dictionary of KNN-based pseudo-labeled DataFrames for each size.
    labeled_splits_zero : Dict[str, pd.DataFrame]
        Dictionary of zero-shot pseudo-labeled DataFrames for each size.
    model : SentenceTransformer
        Model used for generating embeddings.
    weight_orig : float, optional
        Weight for original training data, by default 1.0.
    weight_weak : float, optional
        Weight for weak labels, by default 1.0.
    emb_dir_orig : str, optional
        Directory for original embeddings, by default EMB_DIR_ORIG.
    emb_dir_centroid : str, optional
        Directory for centroid embeddings, by default EMB_DIR_CENTROID.
    emb_dir_zero : str, optional
        Directory for zero-shot embeddings, by default EMB_DIR_ZERO.
    feat_json_path : str, optional
        Path to feature extraction results JSON, by default FEAT_JSON_PATH.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Results for KNN and zero-shot pseudo-labeling experiments.

    Raises
    ------
    FileNotFoundError
        If the feature JSON file is not found.
    """
    try:
        with open(feat_json_path) as f:
            feat_json = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Feature JSON file not found at {feat_json_path}: {e}")

    res_centroid, res_zero = {}, {}
    for size in tqdm(split_data, desc=f"Splits (w_o={weight_orig}, w_w={weight_weak})"):
        lr_params = pick_lr_params(feat_json, size=size)
        split_df = split_data[size].copy()
        split_df["name"] = f"size_{size}"

        # KNN pseudo-labels
        res_centroid[size] = train_with_pseudo_labels(
            split_df, labeled_splits_knn[size], model,
            emb_dir_orig, emb_dir_centroid, lr_params,
            weight_orig=weight_orig, weight_weak=weight_weak
        )

        # Zero-shot pseudo-labels
        res_zero[size] = train_with_pseudo_labels(
            split_df, labeled_splits_zero[size], model,
            emb_dir_orig, emb_dir_zero, lr_params,
            weight_orig=weight_orig, weight_weak=weight_weak
        )

    return res_centroid, res_zero



