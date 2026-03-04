import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from sklearn.neighbors import KNeighborsClassifier

# Hilfsfunktionen
def split_labeled_unlabeled(df):
    df_hard = df[df['label_type'] == 'hard'].copy()
    df_unlabeled = df[df['label_type'] == 'unlabeled'].copy()
    return df_hard, df_unlabeled

def combine_hard_and_weak(df_hard, df_weak):
    df_weak['weak_label'] = df_weak['weak_label'].astype(int)
    return pd.concat([df_hard, df_weak], ignore_index=True)


# CENTROID

def generate_weak_labels_centroid_from_combined(df, text_col='text', embedding_model_name='all-MiniLM-L6-v2'):
    df_hard, df_unlabeled = split_labeled_unlabeled(df)
    if df_hard.empty or df_unlabeled.empty:
        print("Kein hard- oder unlabeled-Daten im Split vorhanden.")
        return df

    model = SentenceTransformer(embedding_model_name)
    embeddings_hard = model.encode(df_hard[text_col].tolist(), show_progress_bar=True)
    embeddings_unlabeled = model.encode(df_unlabeled[text_col].tolist(), show_progress_bar=True)

    labels_hard = df_hard['weak_label'].values
    centroids = {label: np.mean(embeddings_hard[labels_hard == label], axis=0) for label in np.unique(labels_hard)}

    preds = []
    for emb in embeddings_unlabeled:
        sims = {label: cosine_similarity([emb], [centroid])[0][0] for label, centroid in centroids.items()}
        preds.append(int(max(sims, key=sims.get)))

    df_unlabeled['weak_label'] = preds
    df_unlabeled['label_type'] = 'weak'
    return combine_hard_and_weak(df_hard, df_unlabeled)

def apply_centroid_weak_labelling_to_all_splits(semi_splits, text_col='text', embedding_model_name='all-MiniLM-L6-v2'):
    labeled_splits = {}
    for size, df in semi_splits.items():
        print(f"Verarbeite Split {size} mit Centroid-Matching...")
        df_labeled = generate_weak_labels_centroid_from_combined(df, text_col, embedding_model_name)
        labeled_splits[size] = df_labeled
    return labeled_splits


# KNN mit SBERT

def get_dynamic_k(n_hard):
    return max(3, min(10, int(n_hard * 0.01)))

def generate_weak_labels_knn_from_combined(df, text_col='text', embedding_model_name='all-MiniLM-L6-v2', k=5):
    """
    Generiert Weak Labels für unbeschriftete Texte mittels KNN auf Basis von SBERT-Embeddings.
    """
    df_hard, df_unlabeled = split_labeled_unlabeled(df)
    if df_hard.empty or df_unlabeled.empty:
        print("Kein hard- oder unlabeled-Daten im Split vorhanden.")
        return df

    # SBERT laden & Embeddings berechnen
    model = SentenceTransformer(embedding_model_name)
    embeddings_hard = model.encode(df_hard[text_col].tolist(), show_progress_bar=True)
    embeddings_unlabeled = model.encode(df_unlabeled[text_col].tolist(), show_progress_bar=True)

    # KNN-Modell fitten
    y_hard = df_hard['weak_label'].astype(int).values
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(embeddings_hard, y_hard)

    # Vorhersage für unlabelte Daten
    preds = knn.predict(embeddings_unlabeled)

    df_unlabeled = df_unlabeled.copy()
    df_unlabeled['weak_label'] = preds
    df_unlabeled['label_type'] = 'weak'

    return combine_hard_and_weak(df_hard, df_unlabeled)

def apply_knn_weak_labelling_to_all_splits(semi_splits, text_col='text', embedding_model_name='all-MiniLM-L6-v2'):
    labeled_splits = {}
    for size, df in semi_splits.items():
        df_hard, _ = split_labeled_unlabeled(df)
        n_hard = len(df_hard)
        k = get_dynamic_k(n_hard)

        print(f"Verarbeite Split {size} mit KNN-SBERT (k={k}) auf {n_hard} harten Beispielen...")
        df_labeled = generate_weak_labels_knn_from_combined(df, text_col, embedding_model_name, k)
        labeled_splits[size] = df_labeled
    return labeled_splits


# ZERO SHOT

def zero_shot_classify_from_combined(df, text_col='text'):
    df_hard, df_unlabeled = split_labeled_unlabeled(df)
    if df_unlabeled.empty:
        print("Keine unlabelten Daten vorhanden – übersprungen.")
        return df

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["negative", "positive"]
    label_map = {"negative": 0, "positive": 1}

    weak_labels = []
    for text in tqdm(df_unlabeled[text_col], desc="Zero-Shot Klassifikation"):
        output = classifier(text, candidate_labels)
        best_label = output['labels'][0]
        weak_labels.append(label_map[best_label])

    df_unlabeled['weak_label'] = weak_labels
    df_unlabeled['label_type'] = 'weak'
    return combine_hard_and_weak(df_hard, df_unlabeled)

def apply_zero_shot_to_all_splits(semi_splits, text_col='text'):
    labeled_splits = {}
    for size, df in semi_splits.items():
        print(f"Verarbeite Split {size} mit Zero-Shot...")
        df_labeled = zero_shot_classify_from_combined(df, text_col)
        labeled_splits[size] = df_labeled
    return labeled_splits


# RANDOM FOREST CLASS.

def apply_weak_labels_rf_balanced_threshold(df, text_col='text', min_threshold=0.6, max_threshold=0.95, step=0.01):
    """
    Trainiert einen RandomForest auf hart gelabelten Daten und wählt adaptiv einen Schwellenwert,
    bei dem beide Klassen möglichst gleichmäßig im Weak Labeling vertreten sind.
    """
    df_hard, df_unlabeled = split_labeled_unlabeled(df)

    if df_hard.empty or df_unlabeled.empty:
        print("Kein hard- oder unlabeled-Daten im Split vorhanden.")
        return df

    # Text-Vektorisierung
    vec = TfidfVectorizer()
    X_hard = vec.fit_transform(df_hard[text_col])
    y_hard = df_hard['true_label'].astype(int)

    # Modell trainieren
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_hard, y_hard)

    # Vorhersage auf unlabeled Daten
    X_unlabeled = vec.transform(df_unlabeled[text_col])
    probs = clf.predict_proba(X_unlabeled)
    preds = clf.predict(X_unlabeled)
    confidences = probs.max(axis=1)

    # Suche nach bestmöglichem Threshold mit ausgewogener Klassenverteilung
    best_threshold = None
    best_balance = 0
    for threshold in np.arange(min_threshold, max_threshold + step, step):
        accepted_preds = [int(p) if c >= threshold else None for p, c in zip(preds, confidences)]
        accepted = pd.Series(accepted_preds).dropna().astype(int)
        if accepted.empty or len(np.unique(accepted)) < 2:
            continue
        counts = accepted.value_counts(normalize=True)
        balance = 1 - abs(counts.get(0, 0) - counts.get(1, 0))  # maximal bei 0.5/0.5
        if balance > best_balance:
            best_balance = balance
            best_threshold = threshold

    if best_threshold is None:
        print("Kein geeigneter Threshold gefunden – keine Weak Labels gesetzt.")
        return df

    # Anwendung mit optimalem Threshold
    df_unlabeled = df_unlabeled.copy()
    df_unlabeled['weak_label'] = [
        int(p) if c >= best_threshold else None
        for p, c in zip(preds, confidences)
    ]
    df_unlabeled = df_unlabeled[df_unlabeled['weak_label'].notnull()]
    df_unlabeled['weak_label'] = df_unlabeled['weak_label'].astype(int)
    df_unlabeled['label_type'] = 'weak'

    print(f"Optimaler Threshold gewählt: {best_threshold:.2f} | Klassengleichgewicht: {best_balance:.2f}")
    return combine_hard_and_weak(df_hard, df_unlabeled)

def apply_rf_balanced_to_all_splits(semi_splits, text_col='text'):
    """
    Wendet RandomForest mit adaptivem Threshold auf alle Splits an,
    um möglichst ausgewogene Weak Labels zu erzeugen.
    """
    labeled_splits = {}
    for size, df in semi_splits.items():
        print(f"\nVerarbeite Split {size} mit adaptivem RF-Threshold...")
        df_labeled = apply_weak_labels_rf_balanced_threshold(df, text_col=text_col)
        labeled_splits[size] = df_labeled
    return labeled_splits



# HEURISTIC

# Konstanten
POSITIVE = 1
NEGATIVE = 0
ABSTAIN = -1
CARDINALITY = 2  # zwei Klassen: 0 = negativ, 1 = positiv


# Labeling Functions 
# ------------------------------------------

def lf_positive_keywords(text):
    positive_words = ["great", "excellent", "best", "love", "good", "well"]
    return POSITIVE if any(word in text.lower() for word in positive_words) else ABSTAIN

def lf_negative_keywords(text):
    negative_words = ["waste", "bad", "disappointed", "nothing", "worst"]
    return NEGATIVE if any(word in text.lower() for word in negative_words) else ABSTAIN

def lf_sentiment_phrases(text):
    text = text.lower()
    positive_phrases = [
        "absolutely love", "would buy again", "definitely recommend",
        "amazing quality", "great quality", "five stars", "super happy", "not bad"
    ]
    negative_phrases = [
        "very disappointed", "not worth it", "would not recommend",
        "poor quality", "waste of money", "one star", "zero stars", "not good"
    ]
    for phrase in positive_phrases:
        if phrase in text:
            return POSITIVE
    for phrase in negative_phrases:
        if phrase in text:
            return NEGATIVE
    return ABSTAIN

labeling_functions = [lf_positive_keywords, lf_negative_keywords, lf_sentiment_phrases]

# Gewichtung der LFs: [LF1, LF2, LF3]
lf_weights = [0.5, 0.5, 1.0]  # Sentiment-Phrasen als stärkste Quelle


# Gewichtete Majority Vote Logik
# ------------------------------------------

def weighted_majority_vote(pred_matrix: List[List[int]], weights: List[float]) -> List[int]:
    preds = []
    for row in pred_matrix:
        score = {POSITIVE: 0.0, NEGATIVE: 0.0}
        for pred, weight in zip(row, weights):
            if pred == ABSTAIN:
                continue
            score[pred] += weight
        if score[POSITIVE] == 0 and score[NEGATIVE] == 0:
            preds.append(ABSTAIN)
        else:
            preds.append(POSITIVE if score[POSITIVE] >= score[NEGATIVE] else NEGATIVE)
    return preds


# Anwenden der Labeling Functions auf DF
# ------------------------------------------

def apply_lfs_to_dataframe(df: pd.DataFrame, text_col='text') -> pd.DataFrame:
    lf_outputs = []
    for _, row in df.iterrows():
        text = row[text_col]
        row_preds = [lf(text) for lf in labeling_functions]
        lf_outputs.append(row_preds)
    return lf_outputs


# Hauptlogik
# ------------------------------------------

def apply_heuristic_labeling(df, text_col='text'):
    df_hard, df_unlabeled = split_labeled_unlabeled(df)

    if df_unlabeled.empty:
        print("Keine unlabelten Daten vorhanden – übersprungen.")
        return df

    lf_matrix = apply_lfs_to_dataframe(df_unlabeled, text_col)
    weak_preds = weighted_majority_vote(lf_matrix, lf_weights)

    df_unlabeled = df_unlabeled.copy()
    df_unlabeled['weak_label'] = weak_preds
    df_unlabeled['label_type'] = 'weak'

    return combine_hard_and_weak(df_hard, df_unlabeled)


# Batch-Anwendung auf alle Splits
# ------------------------------------------

def apply_heuristics_to_all_splits(semi_splits, text_col='text'):
    """
    Führt heuristisches Weak Labeling für alle Splits aus,
    entfernt ABSTAINs (-1), und meldet deren Anzahl bei Split 1.0.
    """
    labeled_splits = {}
    for size, df in semi_splits.items():
        print(f"Verarbeite Split {size} mit gewichteter Heuristik...")
        df_labeled = apply_heuristic_labeling(df, text_col)

        # Anzahl ABSTAINs berechnen (vor dem Entfernen)
        if size == 1.0:
            n_total = len(df_labeled)
            n_kept = df_labeled['weak_label'].isin([0, 1]).sum()
            n_abstain = n_total - n_kept
            print(f"→ ABSTAIN entfernt in Split 1.0: {n_abstain} von {n_total} ({n_abstain/n_total:.1%})")

        # ABSTAINs entfernen
        df_labeled = df_labeled[df_labeled['weak_label'].isin([0, 1])].copy()
        
        labeled_splits[size] = df_labeled
    return labeled_splits



# Die apply_method_to_all_splits könnte noch zusammengefasst werden, dazu müsste centroid embedding vorher gemacht werden oder separat gelassen werden 
