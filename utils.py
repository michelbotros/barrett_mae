from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scikitplot as skplt
import os
from sklearn.metrics import roc_curve, auc
from matplotlib import colors as mpltcolors

# define some colors
colors_1 = ["white", "green", "orange", "red", 'yellow', 'yellow', 'purple', 'pink', 'grey', "blue"]
colors_2 = ["white", "green", "red", "yellow", 'brown', 'yellow', 'purple', 'pink', 'grey', 'green']
colors_3 = ["green", "green", "red", 'orange', 'orange', 'purple', 'red', 'grey', "blue", 'white']


def plot_confusion_matrix(cf_matrix, save_path=None, pixel_level=True, normalized=True, kappa=None, labels=None):
    """ Plots the confusion matrix

    Args:
        cf_matrix: the confusion matrix to plot
        save_path: location where to store the plot
        plot: whether to plot it of save it
        pixel_level: whether to include bg
        normalized: whether it is normalized
        kappa: plot kappa as title
        labels: the labels of the categories

    Returns:
        none: saves the figure at the save path


    """
    if labels is None:
        if pixel_level:
            labels = ['BG', 'NDBE', 'LGD', 'HGD']
            fmt = '.2f'
        else:
            labels = ['NDBE', 'LGD', 'HGD']
            if normalized:
                fmt = '.2f'
            else:
                fmt = 'd'
    else:
        fmt = 'd'

    df_cm = pd.DataFrame(cf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 22})
    sns.heatmap(df_cm, annot=True, cmap="Blues", square=True, fmt=fmt)
    plt.gca().set_yticklabels(labels=labels, va='center')
    plt.gca().set_ylabel('True', labelpad=30)
    plt.gca().set_xlabel('Predicted', labelpad=30)
    if kappa:
        plt.title('$\kappa=${:.2f}'.format(kappa))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curves(y_true, y_prob, save_path=None, plot_roc_dysplasia=False):
    cmap = mpltcolors.ListedColormap(colors_3)
    skplt.metrics.plot_roc_curve(y_true, y_prob, figsize=(11, 10), text_fontsize=16, title='ROC all classes', cmap=cmap)
    if save_path:
        plt.savefig(os.path.join(save_path, 'test_roc_per_class.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    if plot_roc_dysplasia:

        # get dysplasia probs and targets
        y_true_ndbe_vs_dys = np.where(y_true > 1, 1, 0)
        y_prob_dys = np.expand_dims(y_prob[:, 1] + y_prob[:, 2], axis=1)

        # Compute ROC curve and ROC area for dysplasia class
        fpr, tpr, _ = roc_curve(y_true_ndbe_vs_dys, y_prob_dys)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(11, 10))
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="red",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Dysplasia")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, 'test_roc_dysplasia.png'), bbox_inches='tight')
        plt.close()





