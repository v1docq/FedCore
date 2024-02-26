import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from fedcore.models.backbone.resnet import ResNet


class PerformanceEvaluator:
    def __init__(self, model, dataset, device: str = 'cpu', batch_size=32):
        self.model = model.model if hasattr(model, 'model') else model
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.device = torch.device(device)
        self.model.to(self.device)

        # Measured performance metrics
        self.latency = None
        self.throughput = None
        self.model_size = None

    def eval(self):

        result = dict(latency=self.measure_latency(),
                      throughput=self.measure_throughput(),
                      model_size=self.measure_model_size())
        self.report()
        return result

    def measure_latency(self, reps: int = 50):
        timings = np.zeros((reps, 1))
        if torch.cuda.is_available():
            self.warm_up_cuda()
        with torch.no_grad():
            with tqdm(total=reps, desc='Measuring latency', unit='rep') as pbar:
                for rep in range(reps):
                    for inputs, _ in self.data_loader:
                        start_time = time.time()
                        _ = self.model(inputs)
                        end_time = time.time()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        curr_time = (end_time - start_time) * 1000
                        timings[rep] = curr_time / inputs.size(0)
                        break
                    pbar.update(1)
        self.latency = round(np.mean(timings) / reps, 3)

    def measure_throughput(self, batches: int = 5):
        total_data_size = 0
        start_time = time.time()
        # measure for n batches
        with torch.no_grad():
            with tqdm(total=batches, desc='Measuring throughput', unit='batch') as pbar:
                for inputs, _ in self.data_loader:
                    inputs = inputs.to(self.model.device) if hasattr(self.model, 'device') else inputs
                    if batches == 0:
                        break
                    total_data_size += inputs.size(0)
                    _ = self.model(inputs)
                    batches -= 1
                    pbar.update(1)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        total_time = (time.time() - start_time) / 1000
        self.throughput = round(total_data_size / total_time, 3)

    def measure_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        self.model_size = round(size_all_mb, 3)

    def warm_up_cuda(self, num_iterations=10):
        """Warm up CUDA by performing some dummy computations"""
        if torch.cuda.is_available():
            for _ in range(num_iterations):
                inputs, _ = next(iter(self.data_loader))
                inputs = inputs.to(self.model.device) if hasattr(self.model, 'device') else inputs
                _ = self.model(inputs)

    def report(self):
        print(f"Latency: {self.latency} ms/sample with batch_size {self.batch_size}")
        print(f"Throughput: {self.throughput} samples/s with batch_size {self.batch_size}")
        print(f"Model size: {self.model_size} MB")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, channels=1):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, channels, 224, 224)
        self.targets = torch.randint(0, 10, (num_samples,))
        self.classes = list(range(10))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == "__main__":
    # Example usage:
    # load MNIST dataset
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    # data_set = MNIST(
    #     root='data',
    #     train=True,
    #     download=True,
    #     transform=ToTensor()
    # )
    data_set = DummyDataset(num_samples=1000)
    # define model from dataset configuration
    resnet = ResNet(input_dim=1,
                    output_dim=len(data_set.classes),
                    model_name='ResNet18one')
    evaluator = PerformanceEvaluator(resnet, data_set)
    evaluator.measure_model_size()
    evaluator.measure_latency()
    evaluator.measure_throughput()

    # or
    # performance = evaluator.eval()
    # print(performance)
