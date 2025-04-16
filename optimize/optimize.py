# Ensures this script can be run from project root
import os
import sys

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml
import copy
import torch
import optuna
from torch.utils.data import DataLoader
from optuna.exceptions import TrialPruned

from model.candle_dataset import CandleDataset, split_dataset
from model.squeeze_momentum_transformer import SqueezeMomentumTransformer
from model.training import train_model, evaluate


def load_yaml(path):
    """
    Loads a YAML file from disk.

    Args:
        path (str): Absolute or relative file path.

    Returns:
        dict: Parsed YAML content as dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_from_definition(trial, key, definition):
    """
    Samples a value for a given hyperparameter from its search definition.

    Args:
        trial (optuna.trial.Trial): Current trial object.
        key (str): Hyperparameter name.
        definition (dict): Search space definition.

    Returns:
        Any: Sampled value.
    """
    t = definition["type"]

    if t == "categorical":
        return trial.suggest_categorical(key, definition["values"])

    if t == "int":
        low = int(definition["low"])
        high = int(definition["high"])
        step = definition.get("step")

        if step is not None:
            step = int(step)
            return trial.suggest_int(key, low, high, step=step)
        return trial.suggest_int(key, low, high)

    if t == "float":
        low = float(definition["low"])
        high = float(definition["high"])
        log = definition.get("log", False)
        step = definition.get("step")

        if step is not None:
            step = float(step)
            return trial.suggest_float(key, low, high, log=log, step=step)
        return trial.suggest_float(key, low, high, log=log)

    raise ValueError(f"Unsupported search space type: {t} for key: {key}")


def objective(trial):
    """
    Defines the objective function for Optuna to minimize.

    Args:
        trial (optuna.trial.Trial): Trial object used for suggesting parameters.

    Returns:
        float: Objective value (1 - accuracy).
    """

    # Loads the fixed base model configuration
    base_config = load_yaml(os.path.join(ROOT_DIR, "model", "config.yml"))

    # Loads the optimization search space and fixed settings
    optimize_config = load_yaml(os.path.join(ROOT_DIR, "optimize", "config.yml"))

    # Duplicates the base configuration to apply sampled overrides
    config = copy.deepcopy(base_config)

    # Samples all hyperparameters from the defined search space
    search = optimize_config["search_space"]
    sampled = {}

    for key, definition in search.items():
        sampled[key] = sample_from_definition(trial, key, definition)

    # Expands classifier head definition from n and unit size
    classifier_hidden_layers = [sampled["classifier_hidden_layer"]] * sampled["n_hidden_layers"]

    # Updates the final configuration to be passed to the model and trainer
    config.update({  #
        "model_dimension": sampled["model_dimension"],  #
        "number_of_heads": sampled["number_of_heads"],  #
        "number_of_layers": sampled["number_of_layers"],  #
        "feedforward_dimension": sampled["feedforward_dimension"],  #
        "dropout_rate": sampled["dropout_rate"],  #
        "classifier_hidden_layers": classifier_hidden_layers,  #
        "initializer": sampled["initializer"],  #
        "loss_function": sampled["loss_function"],  #
        "learning_rate": sampled["learning_rate"],  #
        "epochs": sampled["epochs"],  #
        "scheduler": {  #
            "type": sampled["scheduler_type"],  #
            "t_0": 10,  #
            "t_mult": 2,  #
            "eta_min": sampled["eta_min"]  #
        },  #
        "device": optimize_config["device"]  #
    })

    # Loads preprocessed dataset
    dataset = CandleDataset("data/features.npy", "data/labels.npy")
    train_ds, val_ds, _ = split_dataset(dataset)

    # Prepares data loaders
    train_loader = DataLoader(train_ds, batch_size=sampled["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=sampled["batch_size"])

    # Initializes the model
    model = SqueezeMomentumTransformer(config)

    # Reports intermediate metrics for pruning
    def report_callback(epoch, val_loss, val_acc):
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise TrialPruned()

    # Trains the model using current trial's config
    train_model(  #
        model=model,  #
        train_loader=train_loader,  #
        val_loader=val_loader,  #
        test_loader=val_loader,  #
        config=config,  #
        on_epoch_end=report_callback  #
    )

    # Final evaluation on validation set
    _, acc, _, _ = evaluate(  #
        model,  #
        val_loader,  #
        torch.nn.CrossEntropyLoss(),  #
        torch.device(config["device"])  #
    )

    # Returns the score (lower is better)
    score = 1.0 - acc
    print(f"✓ Trial {trial.number} — Accuracy: {acc:.4f} | Score: {score:.4f}")

    return score


if __name__ == "__main__":
    """
    Entry point to launch Optuna optimization.
    """

    # Loads fixed optimization settings
    optimize_config = load_yaml(os.path.join(ROOT_DIR, "optimize", "config.yml"))

    # Creates or resumes an Optuna study
    study = optuna.create_study(  #
        study_name=optimize_config["study_name"],  #
        storage=optimize_config["storage"],  #
        direction="minimize",  #
        load_if_exists=True  #
    )

    # Starts the optimization loop
    study.optimize(  #
        objective,  #
        n_trials=optimize_config["n_trials"],  #
        n_jobs=optimize_config["n_jobs"]  #
    )

    # Final summary
    print("\nBest Trial:")
    print(study.best_trial)
    print("\nBest Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
