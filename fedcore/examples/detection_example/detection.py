import sys
sys.path.append(".")

import torch
import random

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

if __name__ == '__main__':
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    device = default_device()
    
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

    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    num_classes = len(train_dataset.classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)
    model.to(device)
    
    opt = optim.SGD(model.parameters(), lr=4e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True)
    evaluator = PerformanceEvaluatorOD(model, test_loader, batch_size=1)
    
    train_loss = []
    val_loss = []
    
    model.train()   
    # Train the model
    for epoch in range(26):
        model.train()
        running_loss = 0.0
        epoch_loss = []
        for i, (images, targets) in enumerate(train_loader):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step() 
            running_loss += loss.item()
            if i % 50 == 0:    
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            epoch_loss.append(loss)
        if device == 'cuda':
            torch.cuda.empty_cache()
        train_loss.append(epoch_loss.mean())
        model.eval()
        target_metric = evaluator.measure_target_metric()
        print('[%d] MAP: %.3f' %
                    (epoch + 1, target_metric["map"]))
        scheduler.step(target_metric["map"])
        val_loss.append(target_metric["map"])
        if val_loss[-1] == val_loss[-6]:
            print("Early stopping")
            break
    performance = evaluator.eval()
    print('Before quantization')
    print(performance)
    torch.save(model, f'{model._get_name()}_' + DATASET_NAME + '.pt')
    
    model.cpu()
    val_data = val_dataset[random.randint(0, len(val_dataset) - 1)]
    img = val_data[0]
    targets = val_data[1]
    input = torch.unsqueeze(img, dim=0)
    preds = model(input)

    transform = v2.ToPILImage()
    img = transform(img)
    show_image(img, targets, preds, train_dataset.classes)