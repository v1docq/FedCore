import sys
sys.path.append(".")

import os
import torch

from tqdm import tqdm
from torchvision.transforms import v2
    
from fedcore.architecture.dataset.task_specified.object_detection_datasets import YOLODataset, UnlabeledDataset
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.architecture.utils.loader import get_loader
from fedcore.architecture.visualisation.visualization import get_image, apply_nms, filter_boxes


DATASET_NAME = 'chips' # african-wildlife
UNLABELED_DATASET_PATH = f'datasets/{DATASET_NAME}/val/images/'
OUTPUT_PATH = f'datasets/{DATASET_NAME}/output/'

MODEL_NAME = f'FasterRCNN_{DATASET_NAME}_24-07-08.pt'

NMS_THRESH = 0.7 # Intersection-over-Union (IoU) threshold for boxes
THRESH = 0.1 # Score threshold for boxes


if torch.cuda.is_available():
    print("Device:    ", torch.cuda.get_device_name(0))
else:
    print("Device:    CPU")
device = default_device()

if __name__ == '__main__': 
    labeled_dataset = YOLODataset(path=f'datasets/{DATASET_NAME}', dataset_name=DATASET_NAME, log=True) 
    unlabeled_dataset = UnlabeledDataset(images_path=UNLABELED_DATASET_PATH)
    
    loader = get_loader(unlabeled_dataset)

    # Load pretrained model
    model = torch.load(f'{OUTPUT_PATH}{MODEL_NAME}')
    model.to(device)
    
    # Predicting all unlabeled images
    desc = 'Predicting'
    for data in tqdm(loader, desc=desc):
        image = data[0][0]
        name = data[1][0]['name']
        input = torch.unsqueeze(image, dim=0)
        pred = model(input)
        pred = apply_nms(pred[0], NMS_THRESH)
        pred = filter_boxes(pred, THRESH)
        transform = v2.ToPILImage()
        img = transform(image)
        inference_img = get_image(img, pred, labeled_dataset.classes)
        if not os.path.exists(f'{OUTPUT_PATH}images/'):
            os.makedirs(f'{OUTPUT_PATH}images/')
        inference_img.save(f'{OUTPUT_PATH}images/{name}')