# Ensures that the script can be run from any working directory by setting up the root path
import os
import sys

# Compute the root path relative to this file's location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path if not already present
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import optuna


def monitor_optuna_progress(storage_url: str, study_name: str, interval_seconds: int = 30):
    """
    Live terminal monitor for Optuna optimization progress.

    Args:
        storage_url (str): The Optuna database URL.
        study_name (str): The name of the Optuna study.
        interval_seconds (int): Refresh interval in seconds.

    This function continuously prints the progress of the Optuna study,
    including counts of trial states and best trial metrics.
    """

    try:
        # Load the study from the specified storage backend
        study = optuna.load_study(storage=storage_url, study_name=study_name)

        while True:
            # Clear terminal screen for updated output
            os.system("clear" if os.name == "posix" else "cls")

            # Categorize trial statuses
            trials = study.trials
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
            running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
            waiting = [t for t in trials if t.state == optuna.trial.TrialState.WAITING]

            print("=" * 80)
            print(f"Study: {study.study_name}")
            print(f"Total Trials: {len(trials)}")
            print(f"Completed: {len(completed)} | Running: {len(running)} | Waiting: {len(waiting)} | Failed: {len(failed)}")

            if completed:
                best_trial = study.best_trial
                print("\nBest Trial")
                print("-----------")
                print(f"Trial #{best_trial.number}")
                print(f"Score: {best_trial.value:.4f}")
                print("Parameters:")
                for k, v in best_trial.params.items():
                    print(f"  {k}: {v}")
            else:
                print("\nNo successful trials have completed yet.")

            print("=" * 80)
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user.")
    except Exception as e:
        print(f"Error during monitoring: {e}")


if __name__ == "__main__":
    db_path = os.path.join(ROOT_DIR, "optimize", "optimize.db")
    monitor_optuna_progress(storage_url=f"sqlite:///{db_path}", study_name="optimize", interval_seconds=20)
