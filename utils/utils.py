from types import SimpleNamespace
import yaml
from copy import deepcopy
import random
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
import keras
from results_saver import ResultsSaver


def _ns(obj):
    "Recursively convert dicts → SimpleNamespace for dot access."
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def _deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)


def load_config(path) -> SimpleNamespace:
    """
    Load a single YAML file and return a namespace with dot notation.
    """
    path = Path(path).expanduser().resolve()
    with path.open() as fh:
        raw = yaml.safe_load(fh)

    # Propagate num_classes into each softmax layer
    nc = raw["data"]["num_classes"]
    for arch in raw["models"].values():
        if isinstance(arch, dict) and "layers" in arch:
            for layer in arch["layers"]:
                if layer.get("type") == "dense" and layer.get("activation") == "softmax":
                    layer["units"] = nc

    return _ns(raw)


def seed_everything(seed=42):
    """
    **Seed all random number generators for reproducibility.**

    Args:
        seed (int, optional): The seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set Python hash seed


class BestModelCheckpoint(keras.callbacks.Callback):
    """
    Enhanced checkpoint callback that integrates with ResultsSaver.
    """

    def __init__(self, results_saver: ResultsSaver, setting: str, architecture: str,
                 proportion: int, training_phase: str = "", monitor='val_loss', mode='min'):
        super(BestModelCheckpoint, self).__init__()
        self.results_saver = results_saver
        self.setting = setting
        self.architecture = architecture
        self.proportion = proportion
        self.training_phase = training_phase
        self.monitor = monitor
        self.mode = mode
        self.best_value = None
        self.best_epoch = None
        self.checkpoint_path = None

    def on_train_begin(self, logs=None):
        """Initialize checkpoint path."""
        if self.training_phase:
            self.checkpoint_path = self.results_saver.get_model_path(
                self.setting, self.architecture, str(self.training_phase)
            )
        else:
            self.checkpoint_path = self.results_saver.get_model_path(
                self.setting, self.architecture
            )

        if self.training_phase:
            filename = f"{self.training_phase}_proportion_{self.proportion}_best_model.weights.h5"
        else:
            filename = f"proportion_{self.proportion}_best_model.weights.h5"

        self.checkpoint_path = self.checkpoint_path / f'model_checkpoints' / filename

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if self.best_value is None or (self.mode == 'min' and current_value < self.best_value) or (
                self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.model.save_weights(self.checkpoint_path)
