import pandas as pd
from pathlib import Path
import tensorflow as tf
import keras
import numpy as np
from typing import Tuple


class ResultsSaver:
    """
    Handles saving of model checkpoints, training histories, and test results.
    Provides a clean, structured way to save all experiment outputs.
    """

    def __init__(self, base_path: str):
        """
        Initialize ResultsSaver with base experiment path.

        Args:
            base_path: Base directory for saving results
        """

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, setting: str, architecture: str,
                       training_phase: str = "") -> Path:
        """
        Generate standardized path for model-specific results.

        Args:
            setting: Training setting (simple_mixed, fine-tuned)
            architecture: Model architecture name
            training_phase: Optional phase identifier (pretrained, fine_tuned)

        Returns:
            Path object for the model directory
        """
        if training_phase:
            model_dir = self.base_path / setting / architecture / training_phase
        else:
            model_dir = self.base_path / setting / architecture

        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def save_history(self, history: keras.callbacks.History, setting: str,
                     architecture: str, proportion: int, training_phase: str = "") -> None:
        """
        Save training history with metadata.

        Args:
            history: Keras training history
            setting: Training setting
            architecture: Model architecture name
            proportion: Data proportion used
            training_phase: Optional phase identifier

        Returns:
            Path to saved history file
        """
        model_path = self.get_model_path(setting, architecture, training_phase)

        # Create filename
        if training_phase:
            history_name = f"{training_phase}_proportion_{proportion}_history.csv"
        else:
            history_name = f"proportion_{proportion}_history.csv"

        # Save history as CSV
        history_df = pd.DataFrame(history.history)
        history_path = model_path / history_name
        history_df.to_csv(history_path, index=False)

    def save_test_results(self, model: keras.Model, test_dataset: tf.data.Dataset,
                          setting: str, architecture: str, proportion: int,
                          training_phase: str = "") -> None:
        """
        Save detailed test results with predictions and metrics.

        Args:
            model: Trained model
            test_dataset: Test dataset
            setting: Training setting
            architecture: Model architecture name
            proportion: Data proportion used
            training_phase: Optional phase identifier

        """
        model_path = self.get_model_path(setting, architecture, training_phase)

        # Get predictions and true labels
        true_labels, probabilities = self._get_predictions(model, test_dataset)

        # Create results dataframe
        results_df = pd.DataFrame({
            'Predictions': probabilities.tolist(),
            'True Labels': true_labels.tolist(),
        })

        # Save detailed results
        if training_phase:
            results_name = f"{training_phase}_proportion_{proportion}_test_results.csv"
        else:
            results_name = f"proportion_{proportion}_test_results.csv"

        results_path = model_path / results_name

        # Save files
        results_df.to_csv(results_path, index=False)

    @staticmethod
    def _get_predictions(model: keras.Model, test_dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Extract true labels, and probabilities from test dataset."""
        all_true_labels = []
        all_probabilities = []

        for batch in test_dataset:
            inputs, labels = batch

            # Get probabilities
            batch_probabilities = model.predict(inputs, verbose=0)

            all_true_labels.extend(labels)
            all_probabilities.extend(batch_probabilities)

        return np.array(all_true_labels), np.array(all_probabilities)
