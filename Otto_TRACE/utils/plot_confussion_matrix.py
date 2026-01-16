import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(
    cm,
    name_task: str,
    name_classes: list,
    save_path: str | None = None,
    dpi: int = 400
):
    cm = np.asarray(cm)

    # Row normalization 
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    ) * 100

    labels = np.array([
        [f"{int(cm[i, j])}\n({cm_percent[i, j]:.2f}%)"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(5, 4.6))

    sns.heatmap(
        cm_percent,               
        annot=labels,
        fmt="",
        cmap="BuGn",
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar=True,
        vmin=0, vmax=100,         
        cbar_kws={"label": "Percentage (%)"},
        xticklabels=name_classes,
        yticklabels=name_classes,
        ax=ax
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Confusion Matrix – {name_task}\n(Row-normalized: % per true class)",
        pad=12
    )

    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {save_path} (dpi={dpi})")

    return fig
