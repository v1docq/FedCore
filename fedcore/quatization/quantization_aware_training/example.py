import numpy as np
import torch.nn
import torchvision.datasets
from neural_compressor.compression.pruner.utils import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

from fedcore.architecture.utils.paths import data_path
from fedcore.architecture.visualisation.visualization import plot_train_test_loss_metric

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18().to(device)
    torch.save(model.state_dict(), './base_model')
    train_dataset = torchvision.datasets.CIFAR10(data_path('CIFAR10'), train=True, download=True,
                                                 transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    test_dataset = torchvision.datasets.CIFAR10(data_path('CIFAR10'), train=False, download=True,
                                                transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1
    )


    def train_func(model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        num_epochs = 10

        train_losses = []
        train_acces = []
        test_losses = []
        test_acces = []
        total_step = len(train_dataloader)
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}\n')
            model.train()
            running_loss = 0.0
            running_corrects = 0
            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_corrects += torch.sum(preds.detach().cpu() == labels.data)
                if (batch_idx) % 20 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs - 1, batch_idx,
                                                                             total_step, loss.item()))
            scheduler.step()
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects / len(train_dataset)

            train_acces.append(epoch_acc * 100)
            train_losses.append(epoch_loss)

            # evaluation on test
            model.eval()
            test_loss = 0.0
            test_corrects = 0
            for batch_idx, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.to(device))
                _, preds = torch.max(outputs, 1)
                test_loss += loss.item()
                test_corrects += torch.sum(preds.detach().cpu() == labels.data)

            epoch_loss = test_loss / len(test_dataset)
            epoch_acc = test_corrects / len(test_dataset)
            test_acces.append(epoch_acc * 100)
            test_losses.append(epoch_loss)

        return train_losses, test_losses, train_acces, test_acces


    from neural_compressor import QuantizationAwareTrainingConfig
    from neural_compressor.training import prepare_compression

    conf = QuantizationAwareTrainingConfig()
    compression_manager = prepare_compression(model, conf)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model
    train_losses, test_losses, train_acces, test_acces = train_func(model)
    plot_train_test_loss_metric(train_losses, test_losses, train_acces, test_acces)
    model = model.to('cpu')
    compression_manager.callbacks.on_train_end()
    compression_manager.save("./output")
