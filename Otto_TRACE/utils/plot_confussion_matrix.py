import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, name_task : str ,name_classes : int = 2 ):
    """
    Classes are 0 or 1 because its the tasks prediction.
    """
    fig, ax = plt.subplots(fig=(2,2))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                xticklabels=name_classes,
                yticklabels=name_classes,
                ax=ax
                )
    ax.secondary_xaxis("Predicted Classes")
    ax.secondary_xaxis("True Classes")
    ax.set_title(f"Confusion Matrix Single Task for {name_task}")
    return fig