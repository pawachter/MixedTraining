import argparse
import os
from utils.utils import load_config
from utils.utils import seed_everything
from run_experiment import ExperimentRunner

def main():
    ap = argparse.ArgumentParser(description='Run an experiment')
    ap.add_argument("--config", required=True, help="YAML file path")
    ap.add_argument("--seed",   required=True, type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.seed = args.seed                # dynamically attach
    seed_everything(cfg.seed)

    exp_dir = os.path.join("experiments", cfg.experiment, str(cfg.seed))
    cfg.exp_dir = exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    # Run the experiment
    experiment_runner = ExperimentRunner(cfg)
    experiment_runner.run_experiment()

if __name__ == '__main__':
    main()
