import copy
import os

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from model.candle_dataset import CandleDataset, split_dataset
from model.squeeze_momentum_transformer import SqueezeMomentumTransformer
from model.training import train_model, evaluate


def objective(trial):
    with open("model/config.yml", "r") as f:
        base_config = yaml.safe_load(f)
    config = copy.deepcopy(base_config)

    # === Sample core model/training hyperparameters ===
    dim_head_options = {  #
        "d64_h2": (64, 2),  #
        "d64_h4": (64, 4),  #
        "d128_h4": (128, 4),  #
        "d128_h8": (128, 8),  #
        "d256_h4": (256, 4),  #
        "d256_h8": (256, 8),  #
    }
    dim_head_key = trial.suggest_categorical("dim_head_pair", list(dim_head_options.keys()))
    model_dimension, number_of_heads = dim_head_options[dim_head_key]
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    number_of_layers = trial.suggest_int("number_of_layers", 1, 6)
    feedforward_dimension = trial.suggest_int("feedforward_dimension", 64, 512, step=64)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # === Classifier hidden layers ===
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
    classifier_hidden_layers = []
    for i in range(n_hidden_layers):
        layer_size = trial.suggest_int(f"classifier_h{i + 1}", 16, 256, step=16)
        classifier_hidden_layers.append(layer_size)

    # === Scheduler ===
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "cosine_warm_restart", "plateau", "step"])
    t_0 = trial.suggest_int("t_0", 5, 30)
    t_mult = trial.suggest_int("t_mult", 1, 5)
    eta_min = trial.suggest_float("eta_min", 1e-6, 1e-3, log=True)

    # === Early stopping ===
    early_stop_patience = trial.suggest_int("early_stop_patience", 3, 10)
    early_stop_delta = trial.suggest_float("early_stop_delta", 1e-5, 1e-2, log=True)

    # === Labeling config ===
    window_size = trial.suggest_int("window_size", 24, 144)
    prediction_offset = trial.suggest_int("prediction_offset", 1, 12)
    threshold_up = trial.suggest_float("threshold_up", 0.001, 0.01)
    threshold_down = trial.suggest_float("threshold_down", -0.01, -0.001)

    # === Init and loss ===
    initializer = trial.suggest_categorical("initializer", ["xavier", "kaiming"])
    loss_function = trial.suggest_categorical("loss_function", ["cross_entropy", "focal", "label_smoothing"])

    # === Final config override ===
    config.update({"model_dimension": model_dimension, "number_of_heads": number_of_heads, "learning_rate": learning_rate, "dropout_rate": dropout_rate, "number_of_layers": number_of_layers, "feedforward_dimension": feedforward_dimension,
                   "batch_size": batch_size, "classifier_hidden_layers": classifier_hidden_layers, "scheduler": {"type": scheduler_type, "t_0": t_0, "t_mult": t_mult, "eta_min": eta_min, },
                   "early_stopping": {"patience": early_stop_patience, "delta": early_stop_delta, }, "window_size": window_size, "prediction_offset": prediction_offset, "thresholds": {"up": threshold_up, "down": threshold_down, },
                   "initializer": initializer, "loss_function": loss_function, "epochs": 10, "checkpoint_path": "optimize/optuna_temp.pt", })

    # === Data ===
    dataset = CandleDataset("data/features.npy", "data/labels.npy")
    train_ds, val_ds, _ = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # === Logging ===
    print(f"\n{'=' * 60}")
    print(f"→ Starting Trial {trial.number}")
    print("  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"{'=' * 60}", flush=True)

    # === Train ===
    model = SqueezeMomentumTransformer(config)
    train_model(model, train_loader, val_loader, val_loader, config)

    # === Evaluate ===
    _, acc, _, _ = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), torch.device(config["device"]))
    score = 1.0 - acc
    print(f"✓ Trial {trial.number} completed: Accuracy={acc:.4f} | Score={score:.4f}", flush=True)
    return score


if __name__ == "__main__":
    os.makedirs("optimize", exist_ok=True)
    storage_url = "sqlite:///optimize/optimize.db"
    study = optuna.create_study(study_name="optimize", direction="minimize", storage=storage_url, load_if_exists=True, )

    n_trials = 500
    n_jobs = 2

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print("Best trial:")
    print(study.best_trial)
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
