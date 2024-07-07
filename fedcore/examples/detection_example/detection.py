import sys
sys.path.append(".")

import datetime
import torch
import random
import numpy as np

from torch import optim
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn
    
from fedcore.architecture.dataset.object_detection_datasets import YOLODataset
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.tools.ruler import PerformanceEvaluatorOD
from fedcore.architecture.utils.loader import get_loader
from fedcore.architecture.visualisation.visualization import show_image, plot_train_test_loss_metric

DATASET_NAME = 'african-wildlife'
EPOCHS = 35
BATCH_SIZE = 4

if torch.cuda.is_available():
    print("Device:    ", torch.cuda.get_device_name(0))
else:
    print("Device:    CPU")
device = default_device()

if __name__ == '__main__':
    # If dataset doesn't exist, it will be downloaded from 
    # https://docs.ultralytics.com/datasets/detect/#supported-datasets
    # (large datasets like COCO can't be downloaded directly)
    train_dataset = YOLODataset(dataset_name=DATASET_NAME, train=True, log=True)
    val_dataset = YOLODataset(dataset_name=DATASET_NAME, train=False)
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.1, 0.9])
    
    # Define loaders
    train_loader = get_loader(train_dataset, batch_size=BATCH_SIZE, train=True)
    test_loader = get_loader(test_dataset)
    val_loader = get_loader(val_dataset)

    # Define model and number of classes
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    num_classes = len(train_dataset.classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    # Load model
    # model = torch.load('FasterRCNN_african-wildlife.pt')
    
    # Define the optimizer, scheduler and evaluator
    opt = optim.SGD(model.parameters(), lr=4e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, verbose=True)
    tr_evaluator = PerformanceEvaluatorOD(model, train_loader, batch_size=BATCH_SIZE)
    val_evaluator = PerformanceEvaluatorOD(model, test_loader, batch_size=1)
    
    tr_loss = list()
    val_loss = list() 
    tr_map = list()
    val_map = list()
    
    for epoch in range(EPOCHS):
        # Train the model
        model.train()
        
        loss_arr = np.zeros(len(train_loader))
        for i, (images, targets) in enumerate(train_loader):
            # forward
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_arr[i] = loss
            # backward + optimize
            opt.zero_grad()
            loss.backward()
            opt.step()           
        tr_loss.append(loss_arr.mean())
        
        # Calculate train mAP
        model.eval()
        target_metric = tr_evaluator.measure_target_metric()
        tr_map.append(float(target_metric["map"]))
             
        # Evaluate the model
        model.train()
        loss_arr = np.zeros(len(test_loader)) 
        for i, (images, targets) in enumerate(test_loader):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_arr[i] = loss
        val_loss.append(loss_arr.mean())
        
        # Calculate test mAP
        model.eval()
        target_metric = val_evaluator.measure_target_metric()
        val_map.append(float(target_metric["map"]))
        
        # Optimize learning rate
        scheduler.step(float(target_metric["map"]))
        
        # Print metrics
        print('-----------------------------------')
        print('[%d] [TRAIN] Loss: %.3f | mAP: %.3f' %
                (epoch + 1, tr_loss[-1], tr_map[-1]))
        print('[%d] [VAL]   Loss: %.3f | mAP: %.3f' %
                (epoch + 1, val_loss[-1], val_map[-1]))
        
        # Most crucial step
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        if len(val_map) > 5 and val_map[-1] <= val_map[-5]:
            print("Early stopping")
            break
    
    # Final evaluating
    performance = val_evaluator.eval()
    print('Before quantization')
    print(performance)
    
    # Save model
    now = str(datetime.datetime.now())[2:-16]
    torch.save(model, f'{model._get_name()}_{DATASET_NAME}_{now}.pt')
    
    # Inference
    model.cpu()
    id = random.randint(0, len(val_dataset) - 1) # random or int
    val_data = val_loader.dataset[id]
    img, targets = val_data
    input = torch.unsqueeze(img, dim=0)
    preds = model(input)

    # Show inference image
    transform = v2.ToPILImage()
    img = transform(img)
    show_image(img, targets, preds, train_dataset.classes)
    
    # Show metrics graph
    plot_train_test_loss_metric(tr_loss, val_loss, tr_map, val_map)