import sys
sys.path.append(".")
from PIL import Image

import os
import math
import time
import datetime
import torch
import random
import numpy as np
import cv2
from tqdm import tqdm
from torch import optim
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_fpn
    
from fedcore.architecture.dataset.object_detection_datasets import YOLODataset
from fedcore.architecture.comptutaional.devices import default_device
from fedcore.tools.ruler import PerformanceEvaluatorOD
from fedcore.architecture.utils.loader import get_loader
from fedcore.architecture.visualisation.visualization import get_image, plot_train_test_loss_metric, apply_nms, filter_boxes

SKIP_COUNT = 4

NMS_THRESH = 0.5 # Intersection-over-Union (IoU) threshold for boxes
THRESH = 0.1

class FedcoreWrapper:
    def __init__(self, model_save_path: str) -> None:
        self.model_save_path = model_save_path
        if torch.cuda.is_available():
            print("Device:    ", torch.cuda.get_device_name(0))
        else:
            print("Device:    CPU")
        self.device = default_device()

    def init_model(self):
        self.model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    
    def init_data(self, dataset_name: str, batch_size: int = 8):
        self.dataset_name = dataset_name
        self.tr_dataset = YOLODataset(path=f'datasets/{dataset_name}', dataset_name=dataset_name, train=True, log=True)
        self.test_dataset = YOLODataset(path=f'datasets/{dataset_name}', dataset_name=dataset_name, train=False)
        self.batch_size = batch_size
        # Define loaders
        self.tr_loader = get_loader(self.tr_dataset, batch_size=batch_size, train=True)
        self.test_loader = get_loader(self.test_dataset)
    
    
    def train(self, epoch: int = 5, INIT_LR: int = 5e-4):
        num_classes = len(self.tr_dataset.classes)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)
        
        # Define the optimizer and scheduler
        opt = optim.SGD(self.model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=INIT_LR/2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=1, verbose=True)
        
        tr_loss = np.zeros(epoch)
        test_loss = np.zeros(epoch)
        tr_map = np.zeros(epoch)
        test_map = np.zeros(epoch)
        tr_time = np.zeros(epoch)
        
        for ep in range(epoch):       
            tStart = time.time()
            
            # Train the model
            self.model.train()
            loss_arr = np.zeros(len(self.tr_loader))
            desc='Training'
            for i, (images, targets) in enumerate(tqdm(self.tr_loader, desc=desc)):
                # forward
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss_arr[i] = loss
                # backward + optimize
                opt.zero_grad()
                loss.backward()
                opt.step()           
            tr_loss[ep] = loss_arr.mean()
            
            # Calculate train mAP
            self.model.eval()
            evaluator = PerformanceEvaluatorOD(self.model, self.tr_loader, batch_size=self.batch_size)
            target_metric = evaluator.measure_target_metric()
            tr_map[ep] = float(target_metric["map"])
                
            # Evaluate the model
            self.model.train()
            loss_arr = np.zeros(len(self.test_loader)) 
            desc='Evaluating'
            for i, (images, targets) in enumerate(tqdm(self.test_loader, desc=desc)):
                loss_dict = self.model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                loss_arr[i] = loss
            test_loss[ep] = loss_arr.mean()
            
            # Calculate test mAP
            self.model.eval()
            evaluator = PerformanceEvaluatorOD(self.model, self.test_loader, batch_size=1)
            target_metric = evaluator.measure_target_metric()
            test_map[ep] = float(target_metric["map"])
            
            # Optimize learning rate
            scheduler.step(test_map[ep])
            
            # Most crucial step
            if self.device == 'cuda':
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
                model_name = self.model._get_name()
                if not os.path.exists(self.model_save_path):
                    os.makedirs(self.model_save_path)
                torch.save(self.model, f'{self.model_save_path}/{model_name}_{self.dataset_name}.pt')
                skip_count = 0  
            else:
                if test_map[ep] > test_map[:ep].max():
                    print(f'Saving new best model with maP: {test_map[ep]}')
                    torch.save(self.model, f'{self.model_save_path}/{model_name}_{self.dataset_name}.pt')
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
        self.model = torch.load(f'{self.model_save_path}/{model_name}_{self.dataset_name}_{today}.pt')
        evaluator = PerformanceEvaluatorOD(self.model, self.test_loader, batch_size=1)
        performance = evaluator.eval()
        print(performance)
        
        # Show metrics graph
        plot_train_test_loss_metric(tr_loss, test_loss, tr_map, test_map)
        
        # Inference
        self.model.cpu()
        id = random.randint(0, len(self.test_dataset) - 1) # random or int
        test_data = self.test_loader.dataset[id]
        img, target = test_data
        input = torch.unsqueeze(img, dim=0)
        pred = self.model(input)
        pred = apply_nms(pred[0], NMS_THRESH)
        pred = filter_boxes(pred, THRESH)

        # Show inference image
        transform = v2.ToPILImage()
        img = transform(img)
        inference_img = get_image(img, pred, self.tr_dataset.classes, target)
        #inference_img.show()
    
    def load_model(self, model_path: str):
        self.model = torch.load(model_path)
        self.model.to(self.device)
    
    
    
    def predict_one_image(self, path: str = "/run/media/karl/New_SSD/FedCore/datasets/chips/test/images/2873.png", nms_thresh: float = NMS_THRESH, thresh: float = THRESH):
        #img = Image.open("/run/media/karl/New_SSD/FedCore/datasets/chips/test/images/2302.png").convert('RGB')
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.model.cpu()
        transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ])  
        img = transform(img)

        input = torch.unsqueeze(img, dim=0)
        pred = self.model(input)
        pred = apply_nms(pred[0], nms_thresh)
        pred = filter_boxes(pred, thresh)

        # Show inference image
        transform = v2.ToPILImage()
        img = transform(img)
        inference_img = get_image(img, pred, self.tr_dataset.classes)
        open_cv_image = np.array(inference_img)
        #inference_img.show()
        return inference_img
        
    
if __name__ == '__main__':
    fedcore_class = FedcoreWrapper("/run/media/karl/New_SSD/FedCore/fedCore_class")
    fedcore_class.init_data("chips")
    #fedcore_class.init_model()
    
    #fedcore_class.train(epoch=2)
    fedcore_class.load_model("/run/media/karl/New_SSD/FedCore/fedCore_class/FasterRCNN_chips_24-07-08.pt")
    fedcore_class.predict_one_image()