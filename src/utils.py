# utils.py
import numpy as np
import matplotlib.pyplot as plt
import wandb

def one_hot_encode(labels, num_classes):
    """
    Convert an array of labels to one-hot encoded format.
    """
    return np.eye(num_classes)[labels]


def plot_sample_images(X, y, dataset_name="fashion_mnist"):
    """
    Plot one sample image per class in a grid.
    """
    classes = np.unique(y)
    num_classes = len(classes)
    plt.figure(figsize=(10, 4))
    for idx, cls in enumerate(classes):
        img = X[y == cls][0]
        plt.subplot(2, (num_classes + 1) // 2, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Class {cls}")
        plt.axis("off")
    plt.suptitle(f"Sample images from {dataset_name}")
    wandb.log({"sample_images": wandb.Image(plt)})
      
    plt.show()
