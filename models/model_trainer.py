import tensorflow as tf
import keras
from typing import List, Any, Optional, Tuple
from utils.utils import BestModelCheckpoint
from utils.results_saver import ResultsSaver


class ModelTrainer:
    """
    Handles all model training workflows including standard training and fine-tuning.
    Manages callbacks, training history, and different training strategies.
    """

    def __init__(self, training_config: Any, results_saver: ResultsSaver):
        """
        Initialize ModelTrainer with training configuration.

        Args:
            training_config: Configuration object containing training parameters
        """
        self.training_config = training_config
        self.results_saver = results_saver

    def train_model(self, model: keras.Model, train_data: tf.data.Dataset,
                    val_data: tf.data.Dataset, epochs: Optional[int] = None,
                    callbacks: Optional[List[keras.callbacks.Callback]] = None,
                    verbose: int = 2) -> Tuple[keras.Model, keras.callbacks.History, int]:
        """
        Train a model with the given data and parameters.

        Args:
            model: The model to train
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs (uses config default if None)
            callbacks: List of callbacks to use during training
            verbose: Verbosity level for training output

        Returns:
            Tuple of (trained_model, training_history, last_epoch)
        """
        if epochs is None:
            epochs = self.training_config.num_epochs

        if callbacks is None:
            callbacks = []
        # Train the model
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=verbose
        )

        # Calculate last epoch (accounting for early stopping)
        last_epoch = len(history.history['val_loss'])

        return model, history, last_epoch

    def train_simple_mixed(self, model: keras.Model, mixed_data: tf.data.Dataset,
                           val_data: tf.data.Dataset, setting: str, architecture: str, proportion: int,
                           epochs: Optional[int] = None, verbose: int = 2) -> keras.Model:
        """
        Train a model with simple mixed data (real + synthetic combined).

        Args:
            model: The model to train
            mixed_data: Mixed training dataset
            val_data: Validation dataset
            setting: simple_mixed or fine-tuned
            architecture: architecture of the neural network
            proportion: real to synthetic proportion
            epochs: Number of epochs
            verbose: Verbosity level for training output

        Returns:
            Tuple of (trained_model, training_history, last_epoch)
        """
        checkpoint_callback = BestModelCheckpoint(
            results_saver=self.results_saver,
            setting=setting,
            architecture=architecture,
            proportion=proportion,
            monitor='val_loss',
            mode='min'
        )

        callbacks = [checkpoint_callback]

        model, history, _ = self.train_model(
            model=model,
            train_data=mixed_data,
            val_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        # Save training history
        self.results_saver.save_history(
            history=history,
            setting=setting,
            architecture=architecture,
            proportion=proportion
        )

        return model

    def train_with_fine_tuning(self, model: keras.Model, pretrain_data: tf.data.Dataset,
                               finetune_data: tf.data.Dataset, val_data: tf.data.Dataset, setting: str,
                               architecture: str, proportion: int,
                               verbose: int = 2) -> keras.Model:
        """
        Train a model with pretraining followed by fine-tuning.

        Args:
            model: The model to train
            pretrain_data: Pretraining dataset (usually synthetic)
            finetune_data: Fine-tuning dataset (usually real)
            val_data: Validation dataset
            setting: simple_mixed or fine-tuned
            architecture: architecture of the neural network
            proportion: real to synthetic proportion
            verbose: Verbosity level for training output

        Returns:
            Tuple of (trained_model, histories_dict, epochs_dict)
        """

        pretrain_epochs = self.training_config.num_epochs

        # Phase 1: Pretraining
        pretrain_callbacks = [keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=self.training_config.patience,
            verbose=1,
            restore_best_weights=True
        )]

        model, pretrain_history, pretrain_last_epoch = self.train_model(
            model=model,
            train_data=pretrain_data,
            val_data=val_data,
            epochs=pretrain_epochs,
            callbacks=pretrain_callbacks,
            verbose=verbose
        )

        # Save pretraining results
        self.results_saver.save_history(
            history=pretrain_history,
            setting=setting,
            architecture=architecture,
            proportion=proportion,
            training_phase="pretrained"
        )
        # Phase 2: Fine-tuning
        # Calculate remaining epochs for fine-tuning
        finetune_epochs = pretrain_epochs - (pretrain_last_epoch - self.training_config.patience)

        # Callback to save the best performing model
        # Create fine-tuning checkpoint callback
        finetune_checkpoint = BestModelCheckpoint(
            results_saver=self.results_saver,
            setting=setting,
            architecture=architecture,
            proportion=proportion,
            training_phase="fine-tuned",
            monitor='val_loss',
            mode='min'
        )
        finetune_callbacks = [finetune_checkpoint]
        # Clear session to free memory before fine-tuning
        keras.backend.clear_session()

        model, finetune_history, finetune_last_epoch = self.train_model(
            model=model,
            train_data=finetune_data,
            val_data=val_data,
            epochs=finetune_epochs,
            callbacks=finetune_callbacks,
            verbose=verbose
        )

        # Save fine-tuning results
        self.results_saver.save_history(
            history=finetune_history,
            setting=setting,
            architecture=architecture,
            proportion=proportion,
            training_phase="fine-tuned"
        )

        return model
