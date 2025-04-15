import copy
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
    dim_head_options = {
        "d32_h1": (32, 1),
        "d64_h2": (64, 2),
        "d64_h4": (64, 4),
        "d128_h4": (128, 4),
        "d128_h8": (128, 8),
        "d256_h4": (256, 4),
        "d256_h8": (256, 8),
    }

    dim_head_key = trial.suggest_categorical("dim_head_pair", list(dim_head_options.keys()))
    config["model_dimension"], config["number_of_heads"] = dim_head_options[dim_head_key]
    config["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5)
    config["number_of_layers"] = trial.suggest_int("number_of_layers", 1, 6)
    config["feedforward_dimension"] = trial.suggest_int("feedforward_dimension", 64, 512, step=64)
    config["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])

    # === Sample classifier head ===
    hidden_layers = []
    for i in range(trial.suggest_int("n_hidden_layers", 1, 3)):
        hidden_layers.append(trial.suggest_int(f"classifier_h{i + 1}", 16, 256, step=16))
    config["classifier_hidden_layers"] = hidden_layers

    # === Sample scheduler ===
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine", "cosine_warm_restart", "plateau", "step"])
    config["scheduler"] = {
        "type": scheduler_type,
        "t_0": trial.suggest_int("t_0", 5, 30),
        "t_mult": trial.suggest_int("t_mult", 1, 5),
        "eta_min": trial.suggest_float("eta_min", 1e-6, 1e-3, log=True),
    }

    # === Sample early stopping ===
    config["early_stopping"] = {
        "patience": trial.suggest_int("early_stop_patience", 3, 10),
        "delta": trial.suggest_float("early_stop_delta", 1e-5, 1e-2, log=True),
    }

    # === Sample windowing and labeling ===
    config["window_size"] = trial.suggest_int("window_size", 24, 144)
    config["prediction_offset"] = trial.suggest_int("prediction_offset", 1, 12)
    config["thresholds"] = {
        "up": trial.suggest_float("threshold_up", 0.001, 0.01),
        "down": trial.suggest_float("threshold_down", -0.01, -0.001),
    }

    # === Sample initializer + loss function ===
    config["initializer"] = trial.suggest_categorical("initializer", ["xavier", "kaiming"])
    config["loss_function"] = trial.suggest_categorical("loss_function", ["cross_entropy", "focal", "label_smoothing"])

    config["epochs"] = 10
    config["checkpoint_path"] = "optimize/optuna_temp.pt"

    # Data
    dataset = CandleDataset("data/features.npy", "data/labels.npy")
    train_ds, val_ds, _ = split_dataset(dataset)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    print(f"\n{'=' * 60}")
    print(f"→ Starting Trial {trial.number}")
    print("  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"{'=' * 60}", flush=True)

    model = SqueezeMomentumTransformer(config)
    train_model(model, train_loader, val_loader, val_loader, config)

    _, acc, _, _ = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), torch.device(config["device"]))
    score = 1.0 - acc
    print(f"✓ Trial {trial.number} completed: Accuracy={acc:.4f} | Score={score:.4f}", flush=True)
    return score


if __name__ == "__main__":
    storage_url = "sqlite:///optimize.db"
    study = optuna.create_study(
        study_name="optimize",
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
    )

    n_trials = 500
    n_jobs = 6

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    print("Best trial:")
    print(study.best_trial)
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")