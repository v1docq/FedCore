"""Visualization utilities for training curves and object detection outputs.

This module provides helper functions for:

* plotting train / test loss and metric curves over epochs;
* drawing ground-truth and predicted bounding boxes on images;
* applying non-maximum suppression (NMS) to detection outputs;
* filtering detection boxes by score threshold.
"""

import matplotlib.pyplot as plt
from torchvision.ops import nms
from PIL import ImageDraw


colors = [
    "#A5E473",
    "#FF5733",
    "#33A5FF",
    "#FCFF33",
    "#33C4FF",
    "#E033FF",
    "#86FF33",
    "#33FF83",
    "#A5E473",
    "#FF5733",
    "#33A5FF",
    "#FCFF33",
    "#33C4FF",
    "#E033FF",
    "#86FF33",
    "#A5E473",
    "#FF5733",
]


def plot_train_test_loss_metric(train_losses, test_losses, train_metric, test_metric):
    """
    Plot training and validation losses and metrics over epochs.

    Parameters
    ----------
    train_losses : Sequence[float]
        Loss values computed on the training set for each epoch.
    test_losses : Sequence[float]
        Loss values computed on the validation/test set for each epoch.
    train_metric : Sequence[float]
        Metric values (e.g. accuracy, F1) on the training set for each epoch.
    test_metric : Sequence[float]
        Metric values on the validation/test set for each epoch.

    Notes
    -----
    The function creates a figure with two subplots:

    * top: train vs. test loss curves;
    * bottom: train vs. test metric curves.

    The figure is shown via :func:`matplotlib.pyplot.show`.
    """
    fig, axs = plt.subplots(2)

    # Plot the loss curves
    axs[0].plot(train_losses, label="Train Loss")
    axs[0].plot(test_losses, label="Test Loss")
    axs[0].set_title("Loss Curves")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid()

    # Plot the accuracy/metric curves
    axs[1].plot(train_metric, label="Train Metric")
    axs[1].plot(test_metric, label="Test Metric")
    axs[1].set_title("Metric Curves")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Metric")
    axs[1].legend()

    axs[1].grid()

    # Show the plot
    plt.tight_layout()
    plt.show()


def get_image(img, preds, classes, targets=None):
    """
    Draw ground-truth and predicted bounding boxes on an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Image on which bounding boxes will be drawn. The image is modified in-place.
    preds : dict
        Dictionary with prediction results. Expected keys:
        ``"boxes"``, ``"labels"``, ``"scores"`` where:

        * ``boxes`` – tensor with shape (N, 4) in (x1, y1, x2, y2) format;
        * ``labels`` – tensor with class indices;
        * ``scores`` – tensor with confidence scores.
    classes : Sequence[str]
        Class names, where ``classes[label_idx]`` returns the class label string.
    targets : dict, optional
        Dictionary with ground-truth data. Expected keys:
        ``"boxes"`` and ``"labels"`` analogous to prediction, but without scores.
        If provided, target boxes are drawn in red.

    Returns
    -------
    PIL.Image.Image
        The same image object with the drawn bounding boxes and labels.

    Notes
    -----
    * Ground-truth boxes are drawn in red.
    * Predicted boxes are colored according to :data:`colors`.
    * Prediction labels are annotated with class name and score.
    """
    draw = ImageDraw.Draw(img)

    # Target boxes
    if targets is not None:
        for i in range(len(targets["boxes"])):
            x1, y1, x2, y2 = targets["boxes"].detach().numpy()[i]
            draw.rectangle([x1, y1, x2, y2], fill=None, outline="red", width=2)
            label = str(classes[targets["labels"].numpy()[i]])
            draw.text([x1, y1], text=label, fill="red")

    # Prediction boxes
    for i in range(len(preds["boxes"])):
        x1, y1, x2, y2 = preds["boxes"].cpu().detach().numpy()[i]
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=None,
            outline=colors[preds["labels"].cpu().detach().numpy()[i]],
            width=2,
        )
        label = classes[preds["labels"].cpu().numpy()[i]]
        score = preds["scores"].cpu().detach().numpy()[i]
        text = f"{label}: {score:.2f}"
        draw.text([x1 + 5, y2 - 15], text=text, fill="blue")

    return img


def apply_nms(orig_prediction, iou_thresh):
    """
    Apply Non-Maximum Suppression (NMS) to detection predictions.

    Parameters
    ----------
    orig_prediction : dict
        Dictionary with detection outputs, containing keys:
        ``"boxes"``, ``"scores"``, ``"labels"``. All values are expected to be
        PyTorch tensors.
    iou_thresh : float
        IoU threshold used for NMS. Boxes with IoU above this threshold
        (w.r.t. a higher-scoring box) will be suppressed.

    Returns
    -------
    dict
        The same dictionary object with filtered tensors for
        ``"boxes"``, ``"scores"``, and ``"labels"`` (only indices kept by NMS).

    Notes
    -----
    This function modifies ``orig_prediction`` in-place, returning it for
    convenience.
    """
    keep = nms(orig_prediction["boxes"], orig_prediction["scores"], iou_thresh)

    # Keep indices from nms
    final_prediction = orig_prediction
    final_prediction["boxes"] = final_prediction["boxes"][keep]
    final_prediction["scores"] = final_prediction["scores"][keep]
    final_prediction["labels"] = final_prediction["labels"][keep]

    return final_prediction


def filter_boxes(orig_prediction, thresh):
    """
    Filter detection boxes by a score threshold.

    Parameters
    ----------
    orig_prediction : dict
        Dictionary with detection outputs, containing keys:
        ``"boxes"``, ``"scores"``, ``"labels"``. All values are expected to be
        PyTorch tensors.
    thresh : float
        Minimum score value to keep a detection. Predictions with
        ``score <= thresh`` are removed.

    Returns
    -------
    dict
        The same dictionary object with filtered tensors for
        ``"boxes"``, ``"scores"``, and ``"labels"``.

    Notes
    -----
    This function modifies ``orig_prediction`` in-place, returning it for
    convenience.
    """
    keep = orig_prediction["scores"] > thresh

    final_prediction = orig_prediction
    final_prediction["boxes"] = final_prediction["boxes"][keep]
    final_prediction["scores"] = final_prediction["scores"][keep]
    final_prediction["labels"] = final_prediction["labels"][keep]

    return final_prediction
