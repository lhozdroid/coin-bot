import copy

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from model.candle_dataset import CandleDataset, split_dataset
from model.squeeze_momentum_transformer import SqueezeMomentumTransformer
from model.training import train_model, evaluate


def objective(trial):
    # Load and copy base config
    with open("model/config.yml", "r") as f:
        base_config = yaml.safe_load(f)
    config = copy.deepcopy(base_config)

    # Sample hyperparameters
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5)
    config["model_dimension"] = trial.suggest_categorical("model_dimension", [32, 64, 128])
    config["number_of_layers"] = trial.suggest_int("number_of_layers", 1, 4)
    config["number_of_heads"] = trial.suggest_categorical("number_of_heads", [2, 4, 8])
    config["feedforward_dimension"] = trial.suggest_categorical("feedforward_dimension", [64, 128, 256])
    config["classifier_hidden_layers"] = trial.suggest_categorical("classifier_hidden_layers", [[64], [64, 32], [128, 64]])

    # Set short training
    config["epochs"] = 10
    config["checkpoint_path"] = "optuna_temp.pt"

    # Prepare data
    dataset = CandleDataset("data/features.npy", "data/labels.npy")
    train_ds, val_ds, _ = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Initialize and train
    model = SqueezeMomentumTransformer(config)
    train_model(model, train_loader, val_loader, val_loader, config)

    # Final validation
    loss, acc, _, _ = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), torch.device(config["device"]))
    return 1.0 - acc  # We want to minimize the error


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)

    print("Best trial:")
    print(study.best_trial)
    print(study.best_params)
