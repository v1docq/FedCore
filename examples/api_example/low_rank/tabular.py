import sys
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from fedot.core.repository.tasks import (
    Task,
    TaskTypesEnum,
)

from fedcore.api.config_factory import ConfigFactory
from fedcore.api.api_configs import (APIConfigTemplate, AutoMLConfigTemplate, FedotConfigTemplate,
                                     LearningConfigTemplate,
                                       ModelArchitectureConfigTemplate,
                                     TrainingTemplate,
                                       DeviceConfigTemplate, ComputeConfigTemplate,
                                     LowRankTemplate)
# from fedcore.architecture.dataset.api_loader import ApiLoader
# from fedcore.data.dataloader import load_data
from datasets import load_dataset
# from fedcore.tools.example_utils import get_scenario_for_api
from fedcore.api.main import FedCore
from fedcore.api.llm_config import LLMConfigTemplate
from fedcore.data.data import CompressionInputData
# from fedcore.repository.constant_repository import FedotTaskEnum
# from fedcore.metrics.nlp_metrics import NLPAccuracy, NLPF1, SacreBLEU, ROUGE

##########################################################################
### CONFIGURATION ###
##########################################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np

# ==============================
# 1. BIG MLP MODEL ARCHITECTURE
# ==============================
class BigMLPClassifier(nn.Module):
    """
    A large MLP for tabular classification with multiple hidden layers,
    batch normalization, dropout, and residual connections.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int,
        dropout_rate: float = 0.3,
        use_residual: bool = True
    ):
        super(BigMLPClassifier, self).__init__()
        
        self.use_residual = use_residual
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        self.feature_layers = nn.ModuleList(layers)
        
        # Final classification head
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Store dimensions for residual connections
        self.hidden_dims = hidden_dims
        
    def forward(self, x):
        # Apply feature layers with possible residual connections
        for i, layer in enumerate(self.feature_layers):
            # Store input for residual connection
            if self.use_residual and i % 4 == 0:
                residual = x
                
            x = layer(x)
            
            # Add residual connection
            if self.use_residual and i % 4 == 3 and i > 0:
                # Ensure dimensions match for residual connection
                if x.shape[1] == residual.shape[1]:
                    x = x + residual
                    
        return self.classifier(x)

# ==============================
# 2. DATA LOADING AND PREPROCESSING
# ==============================
def load_and_preprocess_data():
    """
    Automatically download the 'pol' dataset from OpenML, preprocess it, 
    and split into train/val/test sets.
    """
    print("Downloading the 'pol' dataset from OpenML...")
    # Fetch the Pol dataset (binary classification, ~10k samples) [citation:1]
    # Using as_frame=True returns a pandas DataFrame for easier preprocessing [citation:12]
    X, y = fetch_openml(data_id=722, as_frame=True, return_X_y=True, parser='auto') # Use data_id for reliability [citation:8]
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Handle missing values if any
    if X.isnull().sum().sum() > 0:
        print("Handling missing values...")
        # For simplicity, fill missing numerical values with median
        # Note: This dataset might have categorical features, but for this example we'll treat all as numeric
        # A more robust solution would handle categorical features separately
        X = X.fillna(X.median())

    # Encode target labels if they are not numeric
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    ohe = OneHotEncoder(sparse_output=False)
    y_encoded = ohe.fit_transform(y_encoded[:, None])
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    
    # Split into train+val and test sets (80/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Split train+val into train and validation sets (70/10 of original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    ) # 0.125 * 0.8 = 0.1, so this gives 70% train, 10% val, 20% test

    print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test)
    
    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, X_train.shape[1], num_classes

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=256):
    """
    Create PyTorch DataLoaders from the preprocessed data.
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, test_dataloader

# ==============================
# 3. MAIN EXECUTION
# ==============================
def prepare():
    # Configuration
    CONFIG = {
        'batch_size': 256,
        'hidden_dims': [512, 256, 128, 64],  # Big MLP architecture
        'dropout_rate': 0.3,
        'use_residual': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {CONFIG['device']}")
    
    # 1. Load and preprocess data (automatically downloaded)
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim, num_classes = load_and_preprocess_data()
    
    # 2. Create DataLoaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        batch_size=CONFIG['batch_size']
    )
    
    # 3. Instantiate model
    model = BigMLPClassifier(
        input_dim=input_dim,
        hidden_dims=CONFIG['hidden_dims'],
        num_classes=num_classes,
        dropout_rate=CONFIG['dropout_rate'],
        use_residual=CONFIG['use_residual']
    ).to(CONFIG['device'])
    
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Return the prepared components
    return model, train_dataloader, val_dataloader, test_dataloader

model, train_loader, val_loader, test_loader = prepare()


example_batch = next(iter(train_loader))
example_input = example_batch[0]  


compression_data = CompressionInputData(
    features=example_input,  
    target=model, 
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader,
    task=Task(TaskTypesEnum.classification),  
    input_dim=example_batch[0].size(-1),
)

################################################################################
### CONFIGURE FEDCORE WITH LLMTrainer AND LOW_RANK PEFT ###
################################################################################

peft_config = LowRankTemplate(
    strategy='explained_variance',
    rank_prune_each=1, 
    custom_criterions=None,
    non_adaptive_threshold=0.7,  
    epochs=10,
    log_each=1,
    eval_each=1,
    decomposer='svd', 
    rank=None,  
    distortion_factor=0.6, 
    random_init='normal',  
    power=3,
)

fedot_config = FedotConfigTemplate(
    problem='classification',
    metric= [
        'BinaryAccuracy',
              'Latency', 
              'ModelSize'
              ],
    pop_size=1,
    timeout=0.1,
    initial_assumption=model
)

device_config = DeviceConfigTemplate(device='cuda' if torch.cuda.is_available() else 'cpu')

automl_config = AutoMLConfigTemplate(fedot_config=fedot_config)


learning_config = LearningConfigTemplate(criterion='cross_entropy',
                                         learning_strategy='from_checkpoint',                                          
                                         peft_strategy_params=[peft_config])

api_template = APIConfigTemplate(automl_config=automl_config,
                                 learning_config=learning_config)

if __name__ == "__main__":
    APIConfig = ConfigFactory.from_template(api_template)
    api_config = APIConfig()
    fedcore_compressor = FedCore(api_config)
    fedcore_compressor.fit(compression_data)
    if hasattr(fedcore_compressor, 'fedcore_model'):
        model_class = fedcore_compressor.fedcore_model.__class__.__name__
        print(f"Trainer: {model_class}")

        if hasattr(fedcore_compressor.fedcore_model, 'operation_impl'):
            trainer_type = type(fedcore_compressor.fedcore_model.operation_impl).__name__
            print(f"Trainer type: {trainer_type}")
    model_comparison = fedcore_compressor.get_report(compression_data)
    print(model_comparison)