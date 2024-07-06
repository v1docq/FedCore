import sys
sys.path.append(".")

import torch
import random
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn
    
from fedcore.architecture.dataset.object_detection_datasets import YOLODataset
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.tools.ruler import PerformanceEvaluatorOD
from fedcore.architecture.utils.loader import collate
from fedcore.architecture.visualisation.visualization import show_image

DATASET_NAME = 'african-wildlife'
EPOCHS = 25

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CPU")

if __name__ == '__main__':
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    device = default_device()
    
    # If dataset doesn't exist, it will be downloaded from 
    # https://docs.ultralytics.com/datasets/detect/#supported-datasets
    # (large datasets like COCO can't be downloaded directly)
    train_dataset = YOLODataset(dataset_name=DATASET_NAME, transform=transform, train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        collate_fn=collate
    )

    val_dataset = YOLODataset(dataset_name=DATASET_NAME, transform=transform, train=False)
    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [0.1, 0.9])
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=collate
    )

    # Define model and number of classes
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    num_classes = len(train_dataset.classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)
    model.to(device)
    
    # Define the optimizer, scheduler and evaluator
    opt = optim.SGD(model.parameters(), lr=4e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, verbose=True)
    evaluator = PerformanceEvaluatorOD(model, test_loader, batch_size=1)
    
    train_loss = list()
    val_loss = list() 
    
    for epoch in range(EPOCHS):
        # Train the model
        model.train()
        running_loss = 0.0
        epoch_loss = np.zeros(len(train_loader))
        
        for i, (images, targets) in enumerate(train_loader):
            # forward
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            # backward + optimize
            opt.zero_grad()
            loss.backward()
            opt.step() 
            
            # Print loss
            running_loss += loss.item()
            if i % 50 == 0:    
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            epoch_loss[i] = loss               
        
        # Evaluate the model
        model.eval()
        target_metric = evaluator.measure_target_metric()
        scheduler.step(float(target_metric["map"]))
        
        print('[%d] MAP: %.3f' %
                    (epoch + 1, target_metric["map"]))
        
        train_loss.append(epoch_loss.mean())
        val_loss.append(float(target_metric["map"]))
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        if len(val_loss) > 5 and val_loss[-1] <= val_loss[-3]:
            print("Early stopping")
            break
    
    # Final evaluating
    performance = evaluator.eval()
    print('Before quantization')
    print(performance)
    torch.save(model, f'{model._get_name()}_' + DATASET_NAME + '.pt')
    
    # Inference
    model.cpu()
    val_data = val_loader.dataset[random.randint(0, len(val_dataset) - 1)]
    img = val_data[0]
    targets = val_data[1]
    input = torch.unsqueeze(img, dim=0)
    preds = model(input)

    # Show inference image
    transform = v2.ToPILImage()
    img = transform(img)
    show_image(img, targets, preds, train_dataset.classes)