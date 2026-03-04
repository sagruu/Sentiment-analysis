import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


#########################
##### Baseline #########
#########################

def baseline_evaluation_plot(json_paths_with_names):
    """
    Vergleicht mehrere Methoden (aus JSON-Dateien) bzgl. Accuracy, Precision, Recall, F1 über Split-Größen hinweg.

    Args:
        json_paths_with_names (list of tuples): [(json_path, method_name), ...]
    
    Returns:
        pd.DataFrame: Alle zusammengeführten Ergebnisse.
    """
    all_records = []

    for json_path, method_name in json_paths_with_names:
        with open(json_path, "r") as f:
            results = json.load(f)

        for size, metrics in results.items():
            try:
                size_val = float(size)
            except ValueError:
                print(f"Ungültiger Split-Name: {size}")
                continue

            # Fine-Tuning-Format?
            if "test" in metrics and isinstance(metrics["test"], dict) and "test_accuracy" in metrics["test"]:
                test = metrics["test"]
                all_records.append({
                    "split_size": size_val,
                    "accuracy": test.get("test_accuracy", 0),
                    "precision": None,  # nicht vorhanden
                    "recall": None,     # nicht vorhanden
                    "f1": test.get("test_f1", 0),
                    "method": method_name
                })

            # Klassisches Format mit classification_report
            else:
                report = metrics.get("classification_report", {})
                weighted = report.get("weighted avg", {})

                all_records.append({
                    "split_size": size_val,
                    "accuracy": metrics.get("test_accuracy", report.get("accuracy", 0)),
                    "precision": weighted.get("precision", 0),
                    "recall": weighted.get("recall", 0),
                    "f1": weighted.get("f1-score", 0),
                    "method": method_name
                })

    df_all = pd.DataFrame(all_records).sort_values(["split_size", "method"])

    # Plot
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 18))

    for i, metric in enumerate(metrics_to_plot):
        for method in df_all["method"].unique():
            df_plot = df_all[df_all["method"] == method]
            axs[i].plot(df_plot["split_size"], df_plot[metric], marker='o', label=method)
        axs[i].set_title(f"{metric.capitalize()} vs Split Size")
        axs[i].set_xlabel("Split Size")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    return df_all


def evaluate_baseline_results(json_path, method_name="TF-IDF"):
    """
    Gibt die numerischen Metriken für ein Baseline-Modell (aus gespeicherten JSON-Ergebnissen) pro Split aus.

    Args:
        json_path (str): Pfad zur JSON-Datei mit Baseline-Ergebnissen.
        method_name (str): Name der Methode (zur Anzeige).
    """
    import json

    print(f"\nEvaluation der Baseline-Ergebnisse – Methode: {method_name.upper()}")
    print("-" * 70)

    with open(json_path, "r") as f:
        results = json.load(f)

    for size in sorted(results.keys(), key=lambda x: float(x)):
        entry = results[size]

        # Prüfe auf Fine-Tuning Struktur
        if "test" in entry and isinstance(entry["test"], dict):
            acc = entry["test"].get("test_accuracy", None)
            f1 = entry["test"].get("test_f1", None)
            precision = entry["test"].get("test_precision", None)
            recall = entry["test"].get("test_recall", None)
        else:
            acc = entry.get("test_accuracy", entry.get("classification_report", {}).get("accuracy", None))
            weighted = entry.get("classification_report", {}).get("weighted avg", {})
            precision = weighted.get("precision", None)
            recall = weighted.get("recall", None)
            f1 = weighted.get("f1-score", None)

        print(f"Split {size}:")
        print(f"Accuracy : {acc:.4f}" if acc is not None else "Accuracy : ---")
        print(f"Precision: {precision:.4f}" if precision is not None else "Precision: ---")
        print(f"Recall   : {recall:.4f}" if recall is not None else "Recall   : ---")
        print(f"F1-Score : {f1:.4f}" if f1 is not None else "F1-Score : ---")
        print("-" * 70)



#########################
##### Embedding #########
#########################

def plot_tsne_for_all_embeddings(splits, prefix, output_path):
    """
    Zeigt t-SNE-Plots für die gegebenen Splits und gespeicherten Embeddings.

    Parameter:
    - splits: Dictionary {size: DataFrame} mit 'label'-Spalte
    - prefix: Präfix der Embedding-Dateien
    - output_path: Speicherort der .npy-Dateien
    """
    for size, subset_df in splits.items():
        embedding_file = os.path.join(output_path, f"{prefix}_embeddings_size_{size}.npy")
        embeddings = np.load(embedding_file)

        labels = subset_df["label"].to_numpy()

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(embeddings)

        pos_points = reduced[labels == 1]
        neg_points = reduced[labels == 0]

        plt.figure(figsize=(10, 8))
        plt.scatter(pos_points[:, 0], pos_points[:, 1], color="blue", label="Positive")
        plt.scatter(neg_points[:, 0], neg_points[:, 1], color="red", label="Negative")
        plt.title(f"t-SNE Plot der Embeddings (Größe={size})")
        plt.xlabel("t-SNE Komponente 1")
        plt.ylabel("t-SNE Komponente 2")
        plt.legend()
        plt.grid(True)
        plt.show()


#########################
##### Weak Labels #######
#########################

def evaluate_weak_labels_quality(labeled_splits, method_name="centroid"):
    """
    Evaluiert die Qualität der Weak Labels gegenüber den true_labels in allen Splits.
    Erwartet: pro Split ein DataFrame mit Spalten: 'weak_label', 'true_label', 'label_type'
    
    Gibt Accuracy, Precision, Recall, F1 für Weak Labels aus.
    """
    print(f"\nEvaluation der Weak Labels – Methode: {method_name.upper()}")
    print("-" * 65)
    
    for size in sorted(labeled_splits.keys()):
        df = labeled_splits[size]
        df_weak = df[df['label_type'] == 'weak'].copy()

        if df_weak.empty or 'true_label' not in df_weak.columns:
            print(f"Kein Vergleich möglich für Split {size} – übersprungen.")
            continue
        
        # Konvertiere Typen robust
        y_true = df_weak['true_label'].astype(int)
        y_pred = df_weak['weak_label'].astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        print(f"Split {size}:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("-" * 65)


def evaluate_and_plot_weak_labels(labeled_splits_centroid, labeled_splits_zero_shot, labeled_splits_knn, labeled_splits_rf, labeled_splits_heuristic):
    """
    Evaluiert die Qualität der Weak Labels  und visualisiert Accuracy, Precision, Recall, F1.
    Erwartet drei Dicts: {split_size: DataFrame mit 'weak_label', 'true_label', 'label_type'}
    """

    def evaluate_splits(labeled_splits, method_name):
        results = []
        for size, df in labeled_splits.items():
            df_weak = df[df['label_type'] == 'weak']
            if df_weak.empty or 'true_label' not in df_weak.columns:
                continue
            y_true = df_weak['true_label'].astype(int)
            y_pred = df_weak['weak_label'].astype(int)
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            results.append({
                'split_size': size,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'method': method_name
            })
        return pd.DataFrame(results)

    # Evaluation durchführen
    df_cent = evaluate_splits(labeled_splits_centroid, "centroid")
    df_zero = evaluate_splits(labeled_splits_zero_shot, "Zero-Shot")
    df_knn = evaluate_splits(labeled_splits_knn, "KNN")
    df_rf = evaluate_splits(labeled_splits_rf, "Random Forest")
    df_heu = evaluate_splits(labeled_splits_heuristic, "Heuristic")

    # Zusammenführen
    df_all = pd.concat([df_cent, df_zero, df_knn, df_rf, df_heu])

    # Plot
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axs = plt.subplots(len(metrics), 1, figsize=(8, 16))

    for i, metric in enumerate(metrics):
        for method in df_all['method'].unique():
            df_plot = df_all[df_all['method'] == method].sort_values('split_size')
            axs[i].plot(df_plot['split_size'], df_plot[metric], marker='o', label=method)
        axs[i].set_title(f"{metric.capitalize()} vs Split Size")
        axs[i].set_xlabel("Split Size")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def inspect_weak_label_distribution(labeled_splits, method_name="centroid"):
    """
    Zeigt für jedes Split:
    - Verteilung der label_type (hard/weak)
    - Verteilung der weak_label-Werte (0/1/NaN)
    - Anzahl final gelabelter Texte
    """
    print(f"\nLabel-Verteilung & Vollständigkeit – Methode: {method_name.upper()}")
    print("=" * 70)

    for size in sorted(labeled_splits.keys()):
        df = labeled_splits[size]
        total = len(df)
        n_missing = df['weak_label'].isna().sum()
        n_labeled = total - n_missing

        print(f"Split {size} – {total} Einträge")
        print("label_type-Verteilung:")
        print(df['label_type'].value_counts(dropna=False).to_string())
        print("\nweak_label-Verteilung:")
        print(df['weak_label'].value_counts(dropna=False).to_string())
        print(f"\nFinal gelabelt: {n_labeled} von {total} | Fehlend: {n_missing}")
        print("-" * 70)


def plot_confusion_matrix_weak_labels(labeled_splits, method_name="centroid", split_size=None, normalize=True):
    """
    Plottet die Confusion Matrix der Weak Labels vs. True Labels.
    Optional für einen bestimmten Split.
    
    Args:
        labeled_splits: dict mit {split_size: DataFrame}
        method_name: Anzeigename im Titel
        split_size: optionaler Split (z. B. 0.2), sonst werden alle kombiniert
        normalize: ob die Matrix normalisiert angezeigt wird
    """
    if split_size:
        dfs = [labeled_splits[split_size]]
        title_suffix = f" (Split {split_size})"
    else:
        dfs = list(labeled_splits.values())
        title_suffix = " (alle Splits)"

    # Daten zusammenführen
    df_all = pd.concat(dfs)
    df_weak = df_all[df_all['label_type'] == 'weak']

    if df_weak.empty:
        print("Keine Weak Labels vorhanden für Confusion Matrix.")
        return

    y_true = df_weak['true_label'].astype(int)
    y_pred = df_weak['weak_label'].astype(int)

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Weak Label Prediction")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix – {method_name}{title_suffix}")
    plt.tight_layout()
    plt.show()


#########################
##### Evaluation ########
#########################

def extract_metrics_per_split(
    results_dict: dict,
    source: str
) -> pd.DataFrame:
    rows = []
    for size, df in results_dict.items():
        df_hard = df[df["label_type"] == "hard"]
        acc = accuracy_score(df_hard["true_label"], df_hard["predicted_label"])
        f1  = f1_score(df_hard["true_label"], df_hard["predicted_label"])
        conf = df_hard["confidence"].mean()
        rows.append({
            "split_size": float(size),
            "accuracy": acc,
            "f1_score": f1,
            "mean_confidence": conf,
            "source": source
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Line-Plot für Accuracy / F1 / Confidence
# ------------------------------------------------------------
def plot_metrics_over_splits(
    metrics_df: pd.DataFrame,
    metrics=("accuracy", "f1_score", "mean_confidence"),
    figsize=(12, 10)
):
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    metrics_df = metrics_df.sort_values("split_size")
    for ax, metric in zip(axes, metrics):
        sns.lineplot(data=metrics_df, x="split_size", y=metric,
                     hue="source", marker="o", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Split-Size")
        ax.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Confusion-Matrix zeichnen
# ------------------------------------------------------------
def plot_confusion_matrix(df: pd.DataFrame, title="Confusion Matrix"):
    cm = confusion_matrix(df["true_label"], df["predicted_label"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ------------------------------------------------------------
# Confidence-Verteilung (korrekt vs falsch)
# ------------------------------------------------------------
def plot_confidence_distribution(df: pd.DataFrame, title="Confidence Distribution"):
    df_hard = df[df["label_type"] == "hard"].copy()
    df_hard["correct"] = df_hard["true_label"] == df_hard["predicted_label"]
    sns.histplot(data=df_hard, x="confidence", hue="correct",
                 bins=30, kde=True, stat="density")
    plt.title(title)
    plt.show()

# ------------------------------------------------------------
# Vollständigen Classification Report ausgeben
# ------------------------------------------------------------
def print_classification_report(df: pd.DataFrame, title="Classification Report"):
    df_hard = df[df["label_type"] == "hard"]
    print(f"===== {title} =====")
    print(classification_report(df_hard["true_label"], df_hard["predicted_label"]))

# ------------------------------------------------------------
# Reports und Plots für gewichtete Experimente
# ------------------------------------------------------------
def full_report_weighted(
    results: dict,
    weight_configs: dict,
    out_dir: str = "./plots_all_splits",
    show: bool = True
):
    """
    Erstellt Metriktabellen, Linienplots, und beschränkt Confusion/Confidence
    auf Centroid und Zero mit weight_weak=0.1.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Metriktabelle + Linienplots für alle Konfigurationen
    all_metrics = []
    for cfg in weight_configs:
        cent_df  = extract_metrics_per_split(results[cfg]['centroid'],  f"Centroid_{cfg}")
        zero_df = extract_metrics_per_split(results[cfg]['zero'], f"Zero_{cfg}")
        all_metrics.extend([cent_df, zero_df])
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_csv = os.path.join(out_dir, "metrics_table_weighted.csv")
    metrics_all.to_csv(metrics_csv, index=False)
    print(f"✔︎ Tabelle aller Splits gespeichert: {metrics_csv}")

    plot_metrics_over_splits(metrics_all)
    lineplot_path = os.path.join(out_dir, "metrics_lineplots_weighted.png")
    plt.savefig(lineplot_path)
    if not show: plt.close()

    # 2) Nur Confusion + Confidence für weight_weak=0.1
    # Bestimme die richtige Config
    selected_cfgs = [k for k,v in weight_configs.items() if v['weight_weak']==0.1]
    if not selected_cfgs:
        raise ValueError("Keine Konfiguration mit weight_weak=0.1 gefunden")
    cfg = selected_cfgs[0]

    for source in ['centroid', 'zero']:
        label = f"{source.upper()}_{cfg}"
        res_dict = results[cfg][source]
        for size, df in res_dict.items():
            # Confusion-Matrix
            fig = plt.figure()
            plot_confusion_matrix(df, title=f"{label} – Split {size}")
            cm_path = os.path.join(out_dir, f"confmat_{label}_{size}.png")
            fig.savefig(cm_path)
            if not show: plt.close(fig)

            # Confidence-Verteilung
            fig = plt.figure()
            plot_confidence_distribution(df, title=f"{label} – Split {size}")
            cd_path = os.path.join(out_dir, f"confhist_{label}_{size}.png")
            fig.savefig(cd_path)
            if not show: plt.close(fig)

            # Classification Report
            print_classification_report(df, title=f"{label} – Split {size}")
            print("-" * 60)

    print(f"Alle Plots & Tabellen liegen in: {os.path.abspath(out_dir)}")
