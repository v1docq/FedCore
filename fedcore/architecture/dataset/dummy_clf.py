class DummyDatasetCLF(torch.utils.data.Dataset):
    def __init__(self, num_samples, channels=1):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, channels, 224, 224)
        self.targets = torch.randint(0, 10, (num_samples,))
        self.classes = list(range(10))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]