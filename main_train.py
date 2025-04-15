import yaml
from torch.utils.data import DataLoader

from model.candle_dataset import CandleDataset, split_dataset
from model.squeeze_momentum_transformer import SqueezeMomentumTransformer
from model.training import train_model

# Load model configuration
with open("model/config.yml", "r") as f:
    config = yaml.safe_load(f)

# Load preprocessed data
dataset = CandleDataset("data/features.npy", "data/labels.npy")
train_ds, val_ds, test_ds = split_dataset(dataset)

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# Initialize the model
model = SqueezeMomentumTransformer(config)

# Train the model
train_model(model, train_loader, val_loader, test_loader, config)
