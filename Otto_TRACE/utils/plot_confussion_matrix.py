import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

def plot_confusion_matrix(
    cm,
    name_task: str,
    name_classes: List[str]):
    """
    Binary confusion matrix for task prediction (0 / 1).
    """
    fig, ax = plt.subplots(figsize=(3, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=name_classes,
        yticklabels=name_classes,
        ax=ax
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix – {name_task}")

    return fig
