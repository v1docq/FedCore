import sys
sys.path.append(".")

import os
import math
import time
import datetime
import torch
import random
import numpy as np

from tqdm import tqdm
from torch import optim
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn
    
from fedcore.architecture.dataset.object_detection_datasets import YOLODataset
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.tools.ruler import PerformanceEvaluatorOD
from fedcore.architecture.utils.loader import get_loader
from fedcore.architecture.visualisation.visualization import get_image, plot_train_test_loss_metric, apply_nms, filter_boxes


DATASET_NAME = 'dataset-5000' # african-wildlife
OUTPUT_PATH = f'datasets/{DATASET_NAME}/output/'

EPS = 50
BATCH_SIZE = 4
INIT_LR = 5e-4
SKIP_COUNT = 4 # max number of epochs w/o mAP improving

NMS_THRESH = 0.5 # Intersection-over-Union (IoU) threshold for boxes
THRESH = 0.3 # Score threshold for boxes


if torch.cuda.is_available():
    print("Device:    ", torch.cuda.get_device_name(0))
else:
    print("Device:    CPU")
device = default_device()

if __name__ == '__main__':
    # If dataset doesn't exist, it will be downloaded from 
    # https://docs.ultralytics.com/datasets/detect/#supported-datasets
    # (large datasets like COCO can't be downloaded directly)
    tr_dataset = YOLODataset(path=f'datasets/{DATASET_NAME}', dataset_name=DATASET_NAME, train=True, log=True)
    test_dataset = YOLODataset(path=f'datasets/{DATASET_NAME}', dataset_name=DATASET_NAME, train=False)
    
    # Define loaders
    tr_loader = get_loader(tr_dataset, batch_size=BATCH_SIZE, train=True)
    test_loader = get_loader(test_dataset)

    # Define model
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # model = torch.load('FasterRCNN_african-wildlife_24-07-07.pt')
    
    # Define model classes
    num_classes = len(tr_dataset.classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    # Define the optimizer and scheduler
    opt = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=INIT_LR/2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=1, verbose=True)
    
    tr_loss = np.zeros(EPS)
    test_loss = np.zeros(EPS)
    tr_map = np.zeros(EPS)
    test_map = np.zeros(EPS)
    tr_time = np.zeros(EPS)
    
    for ep in range(EPS):       
        tStart = time.time()
        
        # Train the model
        model.train()
        loss_arr = np.zeros(len(tr_loader))
        desc='Training'
        for i, (images, targets) in enumerate(tqdm(tr_loader, desc=desc)):
            # forward
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_arr[i] = loss
            # backward + optimize
            opt.zero_grad()
            loss.backward()
            opt.step()           
        tr_loss[ep] = loss_arr.mean()
        
        # Calculate train mAP
        model.eval()
        evaluator = PerformanceEvaluatorOD(model, tr_loader, batch_size=BATCH_SIZE)
        target_metric = evaluator.measure_target_metric()
        tr_map[ep] = float(target_metric["map"])
             
        # Evaluate the model
        model.train()
        loss_arr = np.zeros(len(test_loader)) 
        desc='Evaluating'
        for i, (images, targets) in enumerate(tqdm(test_loader, desc=desc)):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_arr[i] = loss
        test_loss[ep] = loss_arr.mean()
        
        # Calculate test mAP
        model.eval()
        evaluator = PerformanceEvaluatorOD(model, test_loader, batch_size=1)
        target_metric = evaluator.measure_target_metric()
        test_map[ep] = float(target_metric["map"])
        
        # Optimize learning rate
        scheduler.step(test_map[ep])
        
        # Most crucial step
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        # Print metrics
        tEnd = time.time()
        tr_time[ep] = float(tEnd - tStart)
        p = int(math.log(ep + 1, 10))
        print('-' * (40 + p))
        print('| %d | TRAIN | Loss: %.3f | mAP: %.3f |' %
                (ep + 1, tr_loss[ep], tr_map[ep]))
        print('| %d | TEST  | Loss: %.3f | mAP: %.3f |' %
                (ep + 1, test_loss[ep], test_map[ep]))
        print('-' * (13 + p), 
              'Time: %.2f' % tr_time[ep], 
              '-' * 14)
        
        # Saving model
        if ep == 0:
            print(f'Saving baseline model with maP: {test_map[ep]}')
            today = str(datetime.datetime.now())[2:-16]
            model_name = model._get_name()
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            torch.save(model, f'{OUTPUT_PATH}{model_name}_{DATASET_NAME}_{today}.pt')
            skip_count = 0  
        else:
            if test_map[ep] > test_map[:ep].max():
                print(f'Saving new best model with maP: {test_map[ep]}')
                torch.save(model, f'{OUTPUT_PATH}{model_name}_{DATASET_NAME}_{today}.pt')
            else:
                skip_count += 1
                print('Skip model due lower mAP, skip count: ', skip_count)
                
                # Early stop
                if skip_count == SKIP_COUNT:
                    tr_loss = tr_loss[:ep + 1]
                    test_loss = test_loss[:ep + 1]
                    tr_map = tr_map[:ep + 1]
                    test_map = test_map[:ep + 1]
                    train_time = tr_time[:ep + 1]
                    print('Early stopping')
                    break
    
    # Final evaluating
    model = torch.load(f'{OUTPUT_PATH}{model_name}_{DATASET_NAME}_{today}.pt')
    evaluator = PerformanceEvaluatorOD(model, test_loader, batch_size=1)
    performance = evaluator.eval()
    print(performance)
    
    # Show metrics graph
    plot_train_test_loss_metric(tr_loss, test_loss, tr_map, test_map)
    
    # Inference
    model.cpu()
    id = random.randint(0, len(test_dataset) - 1) # random or int
    test_data = test_loader.dataset[id]
    img, target = test_data
    input = torch.unsqueeze(img, dim=0)
    pred = model(input)
    pred = apply_nms(pred[0], NMS_THRESH)
    pred = filter_boxes(pred, THRESH)

    # Show inference image
    transform = v2.ToPILImage()
    img = transform(img)
    inference_img = get_image(img, pred, tr_dataset.classes, target)
    inference_img.show()