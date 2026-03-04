import os
import re
import nbformat
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.display import display, Markdown



def create_semi_supervised_splits_ratio_combined(nested_splits, label_col='label', label_ratio=0.2, seed=42):
    """
    Erstellt für jeden Split einen kombinierten DataFrame mit 'weak_label', 'label_type' und 'true_label'.
    """
    semi_splits = {}

    for size, df in nested_splits.items():
        n_total = len(df)
        n_labeled = int(n_total * label_ratio)

        if n_labeled < 1:
            print(f"Split {size} enthält zu wenige Beispiele – übersprungen.")
            continue

        try:
            df_hard, df_unlabeled = train_test_split(
                df,
                train_size=label_ratio,
                stratify=df[label_col],
                random_state=seed
            )
        except ValueError:
            df_hard, df_unlabeled = train_test_split(
                df,
                train_size=label_ratio,
                random_state=seed,
                shuffle=True
            )

        # Hard-labeled Teil
        df_hard = df_hard.copy()
        df_hard['weak_label'] = df_hard[label_col]
        df_hard['label_type'] = 'hard'
        df_hard['true_label'] = df_hard[label_col]

        # Unlabeled Teil
        df_unlabeled = df_unlabeled.copy()
        df_unlabeled['weak_label'] = None
        df_unlabeled['label_type'] = 'unlabeled'
        df_unlabeled['true_label'] = df_unlabeled[label_col]
        df_unlabeled = df_unlabeled.drop(columns=[label_col])

        # Kombinieren
        df_combined = pd.concat([df_hard, df_unlabeled], ignore_index=True)

        semi_splits[size] = df_combined

    return semi_splits


# Automatisches Inhaltsverzeichnis für Jupyter Notebooks
def generate_toc(nb_path: str = None) -> None:
    """
    Generates and displays a Markdown-based table of contents (TOC) for a Jupyter Notebook.

    This function parses the notebook file, extracts Markdown headers (`#` to `######`),
    and constructs a navigable TOC using anchor links compatible with Jupyter's rendering.

    Parameters:
        nb_path (str, optional): Path to the `.ipynb` file. If None, attempts to detect
                                 the current notebook path using `ipynbname`.

    Side Effects:
        Displays the table of contents inline using IPython's Markdown display.

    Notes:
        - Requires the `ipynbname` package for automatic path detection.
        - Anchors are created by lowercasing header text, stripping punctuation,
          and replacing spaces with hyphens.
        - Headers of level 2 (`##`) and deeper are indented accordingly.

    Example:
        >>> generate_toc()  # Automatically detects current notebook
        >>> generate_toc("notebooks/example.ipynb")  # For a specific file
    """
    
    if nb_path is None:
        try:
            nb_path = ipynbname.path()
        except Exception:
            display(Markdown("*Fehler: Notebook-Dateipfad konnte nicht erkannt werden.*"))
            return
    
    with open(nb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    toc_lines = ["#Inhaltsverzeichnis\n"]
    
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                match = re.match(r'^(#{1,6})\s+(.*)', line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    anchor = re.sub(r'[^\w\s-]', '', title).replace(' ', '-')
                    anchor = anchor.lower()
                    indent = '  ' * (level - 1)
                    toc_lines.append(f"{indent}- [{title}](#{anchor})")

    toc_md = "\n".join(toc_lines)
    display(Markdown(toc_md))