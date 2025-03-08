# utils.py
import numpy as np
import matplotlib.pyplot as plt
import wandb
import plotly.graph_objects as go

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

def plot_confusion_matrix(conf_matrix, labels, run_name="Confusion Matrix", project="DA6401_Assignment1"):
    """
    Logs an interactive confusion matrix to Weights & Biases using Plotly.
    
    Two heatmap traces are created:
      - One for correct (diagonal) predictions with detailed hover info.
      - One for misclassifications (off-diagonals) with a custom hover message.
    
    Two dropdown menus are provided:
      1. "View Mode": Toggle between "Show All", "Only Correct", or "Only Misclassifications".
      2. "Class Focus": Select a specific class (or "All Classes") to focus on. In focus mode, only cells
         in the selected class’s row or column are shown, and an annotation appears summarizing that class's metrics.
         When focusing, the colorbars are hidden.
    
    Parameters:
      conf_matrix (2D array-like): The confusion matrix.
      labels (list): List of class labels.
      run_name (str): Wandb run name.
      project (str): Wandb project name.
    """
    # Convert input to float NumPy array (to allow NaN)
    conf_matrix = np.array(conf_matrix, dtype=float)
    num_classes = conf_matrix.shape[0]
    
    # Calculate per-class metrics (Recall, Precision, F1)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    recalls = np.zeros(num_classes)
    precisions = np.zeros(num_classes)
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        recalls[i] = conf_matrix[i, i] / row_sums[i] if row_sums[i] > 0 else 0.0
        col_sum = conf_matrix[:, i].sum()
        precisions[i] = conf_matrix[i, i] / col_sum if col_sum > 0 else 0.0
        if recalls[i] + precisions[i] > 0:
            f1_scores[i] = 2 * recalls[i] * precisions[i] / (recalls[i] + precisions[i])
        else:
            f1_scores[i] = 0.0

    # Create separate matrices for correct predictions (diagonals) and misclassifications (off-diagonals)
    diag_matrix = np.full(conf_matrix.shape, np.nan)
    mis_matrix = conf_matrix.copy()
    for i in range(num_classes):
        diag_matrix[i, i] = conf_matrix[i, i]
        mis_matrix[i, i] = np.nan

    # Build customdata for the diagonal cells (for hover display of metrics)
    customdata = np.empty((num_classes, num_classes, 3))
    customdata[:] = np.nan
    for i in range(num_classes):
        customdata[i, i, 0] = recalls[i]
        customdata[i, i, 1] = precisions[i]
        customdata[i, i, 2] = f1_scores[i]

    # Heatmap for correct predictions with detailed hover info
    trace_diag = go.Heatmap(
        z=diag_matrix.tolist(),
        x=labels,
        y=labels,
        customdata=customdata.tolist(),
        hoverongaps=False,
        hovertemplate=(
            "Predicted %{z} %{y}s just right<br>" +
            "Recall: %{customdata[0]:.2f}<br>" +
            "Precision: %{customdata[1]:.2f}<br>" +
            "F1: %{customdata[2]:.2f}<extra></extra>"
        ),
        coloraxis="coloraxis2",  # Green color axis for correct predictions
        showscale=True,
        name="Correct Predictions"
    )
    
    # Heatmap for misclassifications with custom hover text
    trace_mis = go.Heatmap(
        z=mis_matrix.tolist(),
        x=labels,
        y=labels,
        hoverongaps=False,
        hovertemplate="Predicted %{z} %{x}s as %{y}s<br><extra></extra>",
        coloraxis="coloraxis",  # Red color axis for misclassifications
        showscale=True,
        name="Incorrect"
    )
    
    # Create base figure with both traces
    fig = go.Figure(data=[trace_diag, trace_mis])
    
    # Update layout (axes, colorbars, and background)
    fig.update_layout(
        title={"text": "Confusion Matrix", "x": 0.5},
        width=750,
        height=600,
        xaxis=dict(
            title="True Label",
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            showgrid=False
        ),
        yaxis=dict(
            title="Predicted Label",
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=labels,
            showgrid=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Misclassification color axis (red)
        coloraxis=dict(
            colorscale=[[0, "rgba(180, 0, 0, 0.05)"], [1, "rgba(180, 0, 0, 0.58)"]],
            colorbar=dict(title="Incorrect", x=1.00)
        ),
        # Correct predictions color axis (green)
        coloraxis2=dict(
            colorscale=[[0, "rgba(0, 180, 0, 0.44)"], [1, "rgba(0, 180, 0, 1)"]],
            colorbar=dict(title="Correct Predictions", x=1.12)
        )
    )
    
    # "View Mode" dropdown: Toggle full matrix, only correct, or only misclassifications.
    view_mode_buttons = [
        dict(
            label="Show All",
            method="update",
            args=[{"visible": [True, True]}, 
                  {"coloraxis.colorbar.visible": True,
                   "coloraxis2.colorbar.visible": True}]
        ),
        dict(
            label="Only Correct",
            method="update",
            args=[{"visible": [True, False]},
                  {"coloraxis.colorbar.visible": False,
                   "coloraxis2.colorbar.visible": True}]
        ),
        dict(
            label="Only Misclassifications",
            method="update",
            args=[{"visible": [False, True]},
                  {"coloraxis.colorbar.visible": True,
                   "coloraxis2.colorbar.visible": False}]
        )
    ]
    
    # "Class Focus" dropdown: Select a class to focus on or "All Classes" for full view.
    focus_buttons = []
    focus_buttons.append(dict(
        label="All Classes",
        method="update",
        args=[{"z": [diag_matrix.tolist(), mis_matrix.tolist()]},
              {"annotations": []}]
    ))
    
    # For each class, mask the matrix so only the row or column for that class remains.
    for idx, lab in enumerate(labels):
        mask = np.zeros(conf_matrix.shape, dtype=bool)
        indices = np.indices(conf_matrix.shape)
        mask[(indices[0] == idx) | (indices[1] == idx)] = True
        
        diag_focus = np.where(mask, diag_matrix, np.nan)
        mis_focus = np.where(mask, mis_matrix, np.nan)
        
        # Annotation for selected class (showing class name and metrics)
        ann_text = (
            f"<b>{lab}</b><br>"
            f"Recall: {recalls[idx]*100:.1f}%<br>"
            f"Precision: {precisions[idx]*100:.1f}%<br>"
            f"F1: {f1_scores[idx]*100:.1f}%"
        )
        annotation = dict(
            xref="paper", yref="paper",
            x=0.02, y=0.95,
            text=ann_text,
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)"
        )
        
        focus_buttons.append(dict(
            label=lab,
            method="update",
            args=[{"z": [diag_focus.tolist(), mis_focus.tolist()]},
                  {"annotations": [annotation],
                   "coloraxis.colorbar.visible": False,
                   "coloraxis2.colorbar.visible": False}]
        ))
    
    # Add dropdown menus with opaque backgrounds.
    fig.update_layout(
        updatemenus=[
            {
                "buttons": view_mode_buttons,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.02,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "rgba(200,200,200,0.9)",
                "bordercolor": "black",
                "borderwidth": 1
            },
            {
                "buttons": focus_buttons,
                "direction": "down",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.95,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
                "bgcolor": "rgba(200,200,200,0.9)",
                "bordercolor": "black",
                "borderwidth": 1
            }
        ]
    )
    
    # Log the figure to wandb
    wandb.init(project=project, name=run_name)
    wandb.log({"confusion_matrix": wandb.Plotly(fig)})
    wandb.finish()
