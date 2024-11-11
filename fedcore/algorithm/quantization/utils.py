import torch

from fedcore.models.network_impl.layers import IDecomposed

def _recreate_embedding(module):
        assert isinstance(module, torch.nn.Embedding)
        new = torch.nn.Embedding(
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.sparse,
            module.weight,
            getattr(module, 'freeze', None),
            getattr(module.weight, 'device', torch.device('cpu')),
            getattr(module.weight, 'dtype', torch.float32)
        )
        return new

def _recreate_linear(module):
    assert isinstance(module, torch.nn.Linear)
    raise NotImplementedError


class ParentalReassembler:    
    supported_layers = {torch.nn.Embedding: _recreate_embedding,
                        torch.nn.Linear: _recreate_linear}
    
    @classmethod
    def set_layer(cls, model, name, module):
        ### TODO check Sequential
        *path, name = name.split('.')
        current_module = model
        for child in path:
            current_module = getattr(current_module, child)
        setattr(current_module, name, module)
            
    
    @classmethod
    def fetch_module(cls, module):
        is_decomposed = isinstance(module, IDecomposed)
        for supported in cls.supported_layers:
            if isinstance(module, supported):
                return supported, is_decomposed
        return None, is_decomposed
    
    @staticmethod
    def decomposed_handle(*args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def handle(cls, module, type):
        return cls.supported_layers[type](module)

    @classmethod
    def convert(cls, module):
        associated, is_decomp = cls.fetch_module(module)
        if associated is None:
            print('failed to fetch:', module.__class__.__name__)
            return None
        if is_decomp:
            new_module = cls.decomposed_handle(module)
        else:
            new_module = cls.handle(module, associated)
        return new_module
    
    @classmethod
    def reassemble(cls, model):
        for name, module in model.named_modules():
            new_module = cls.convert(module)
            if new_module is None:
                continue
            cls.set_layer(model, name, new_module)
        return model
