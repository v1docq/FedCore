
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

#
# _PALETTE = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0),
#             (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0,
#                                                         128), (128, 0, 0), (220, 20, 60), (255, 165, 0),
#             (218, 165, 32), (240, 230, 140), (154, 205,
#                                               50), (107, 142, 35), (0, 100, 0), (46, 139, 87),
#             (32, 178, 170), (64, 224, 208), (70, 130, 180), (138,
#                                                              43, 226), (72, 61, 139), (147, 112, 219),
#             (139, 0, 139), (218, 112, 214), (219, 112,
#                                              147), (255, 20, 147), (255, 228, 196), (139, 69, 19),
#             (210, 105, 30), (244, 164, 96), (188, 143, 143), (112, 128, 144), (230, 230, 250), (245, 245, 245))
#
#
# def _2nparray(arrs: List[Union[torch.Tensor, List]]) -> List[np.ndarray]:
#     return [arr.numpy() if isinstance(arr, torch.Tensor) else np.array(arr) for arr in arrs]
#
#
# def draw_sample_with_bboxes(
#         image: Union[torch.Tensor, str],
#         target: Optional[Dict[str, Union[torch.Tensor, List]]] = None,
#         prediction: Optional[Dict[str, Union[torch.Tensor, List]]] = None,
#         threshold: float = 0.5
# ) -> plt.Figure:
#     """
#     Returns the image with bounding boxes.
#
#     Args:
#         image: image tensor or path to image.
#         target: Dictionary of target values with keys ``'boxes'`` and ``'labels'``.
#         prediction: Dictionary of predicted values with keys ``'boxes'``, ``'labels'`` and ``'scores'``.
#         threshold: Confidence threshold for displaying predicted bounding boxes.
#
#     Returns:
#         `matplotlib.pyplot.Figure` of the image with bounding boxes.
#
#     """
#     assert prediction is not None or target is not None, "At least one parameter from 'target' and 'prediction' must not be None"
#
#     if isinstance(image, torch.Tensor):
#         image = image.permute(1, 2, 0).numpy()
#     else:
#         image = plt.imread(image)
#
#     n = 1 if prediction is None or target is None else 2
#     fig = plt.figure(figsize=(10 * n, 10))
#
#     thickness = 1 + int(image.shape[-2] / 500)
#     font_scale = image.shape[-2] / 1000
#
#     if target is not None:
#         ax = plt.subplot(1, n, 1)
#         boxes, labels = _2nparray([target['boxes'], target['labels']])
#         timage = image.copy()
#         for box, label in zip(boxes.astype(np.int32), labels.astype(str)):
#             cv2.rectangle(timage, (box[0], box[1]),
#                           (box[2], box[3]), (220, 255, 255), thickness)
#             cv2.putText(timage, label, (box[0], box[1]),
#                         0, font_scale, (255, 255, 255), thickness)
#         ax.set_axis_off()
#         ax.imshow(timage)
#
#     if prediction is not None:
#         ax = plt.subplot(1, n, n)
#         boxes, labels, scores = _2nparray(
#             [prediction['boxes'], prediction['labels'], prediction['scores']])
#
#         not_thresh = scores > threshold
#         boxes = boxes[not_thresh]
#         labels = labels[not_thresh]
#         scores = scores[not_thresh]
#
#         pimage = image.copy()
#         for box, label, score in zip(boxes.astype(np.int32), labels, scores):
#             cv2.rectangle(pimage, (box[0], box[1]),
#                           (box[2], box[3]), (220, 255, 255), thickness)
#             cv2.putText(pimage, f'{label} ({score:.2f})',
#                         (box[0], box[1]), 0, font_scale, (255, 255, 255), thickness)
#         ax.set_axis_off()
#         ax.imshow(pimage)
#     return fig
#
#
# def _put_mask(axis: plt.axis, image: np.ndarray, mask: np.ndarray, palette: Tuple):
#     thickness = 1 + int(image.shape[-2] / 500)
#     image = image.copy()
#     for ch in range(mask.shape[0]):
#         contours, _ = cv2.findContours(
#             mask[ch, :, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#         for contour in contours:
#             cv2.polylines(image, contour, True, palette[ch], thickness)
#     axis.set_axis_off()
#     axis.imshow(image)
#
#
# def draw_sample_with_masks(
#         image: torch.Tensor,
#         target: torch.Tensor = None,
#         prediction: torch.Tensor = None,
#         palette: Tuple = _PALETTE
# ) -> plt.Figure:
#     """
#     Returns the image with masks.
#
#     Args:
#         image: Image tensor.
#         target: N-channel target tensor with masks, where n - number of classes.
#         prediction: N-channel prediction tensor with masks, where n - number of classes.
#         palette: Color palette for each class.
#
#     Returns:
#         `matplotlib.pyplot.Figure` of the image with masks.
#
#     """
#     assert prediction is not None or target is not None, "At least one parameter from 'target' and 'prediction' must not be None"
#
#     image = image.permute(1, 2, 0).numpy()
#
#     n = 1 if prediction is None or target is None else 2
#     fig = plt.figure(figsize=(10 * n, 10))
#
#     if target is not None:
#         ax = plt.subplot(1, n, 1)
#         _put_mask(ax, image, target.numpy().astype(np.uint8), palette)
#
#     if prediction is not None:
#         ax = plt.subplot(1, n, n)
#         _put_mask(ax, image, prediction.numpy().astype(np.uint8), palette)
#     return fig


def plot_train_test_loss_metric(train_losses, test_losses, train_metric, test_metric):
    """
    Plots train and test losses and metric by epochs

    :param train_losses:
    :param test_losses:
    :param train_metric:
    :param test_metric:
    :return:
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

    # Plot the accuracy curves
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

    keep = nms(orig_prediction["boxes"], orig_prediction["scores"], iou_thresh)

    # Keep indices from nms
    final_prediction = orig_prediction
    final_prediction["boxes"] = final_prediction["boxes"][keep]
    final_prediction["scores"] = final_prediction["scores"][keep]
    final_prediction["labels"] = final_prediction["labels"][keep]

    return final_prediction


def filter_boxes(orig_prediction, thresh):

    keep = orig_prediction["scores"] > thresh

    final_prediction = orig_prediction
    final_prediction["boxes"] = final_prediction["boxes"][keep]
    final_prediction["scores"] = final_prediction["scores"][keep]
    final_prediction["labels"] = final_prediction["labels"][keep]

    return final_prediction
