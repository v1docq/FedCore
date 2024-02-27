from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

from fedcore.architecture.comptutaional.devices import default_device

MODEL_ZOO = {
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
}


class DenseNetwork:
    def __init__(
            self,
            input_dim: int = 3,
            output_dim: int = 2,
            model_name: str = 'densenet121',
            **kwargs
    ):
        if model_name not in MODEL_ZOO:
            raise ValueError(f'Unknown model name: {model_name}. Available models: {list(MODEL_ZOO.keys())}')
        self.model = MODEL_ZOO[model_name](
            num_init_features=input_dim,
            num_classes=output_dim,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, x):
        x = x.to(default_device())
        return self.model(x)
