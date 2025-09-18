import keras
import gc
from typing import List, Any
from types import SimpleNamespace
from models.networks.cnn import AlexNet
from models.networks.mlp import MLP
from models.networks.vit import ViT


class ModelBuilder:
    """
    Factory class for creating and configuring machine learning models.
    Handles model instantiation and compilation for different architectures.
    """

    def __init__(self, model_configs: SimpleNamespace, data_configs: SimpleNamespace, compile_config: SimpleNamespace):
        """
        Initialize ModelFactory with model and compilation configurations.

        Args:
            model_configs: Dictionary containing model architecture configurations
            compile_config: Configuration object containing compilation parameters
        """
        self.model_configs = model_configs
        self.data_configs = data_configs
        self.compile_config = compile_config
        self._available_models = {
            'cnn': AlexNet,
            'mlp': MLP,
            'vit': ViT
        }

    def create_model(self, architecture_name: str, compile_model: bool = True) -> keras.Model:
        """
        Create a model with the specified architecture.

        Args:
            architecture_name: Name of the architecture ('cnn', 'mlp', 'vit')
            compile_model: Whether to compile the model (default: True)

        Returns:
            Created (and optionally compiled) Keras model

        Raises:
            ValueError: If architecture_name is not supported
            KeyError: If architecture configuration is missing
        """
        if architecture_name not in self._available_models:
            available = list(self._available_models.keys())
            raise ValueError(f"Unsupported architecture '{architecture_name}'. "
                             f"Available: {available}")

        if not hasattr(self.model_configs, architecture_name):
            raise KeyError(f"Configuration for '{architecture_name}' not found in model_configs")

        # Get the model class and configuration
        model_class = self._available_models[architecture_name]
        model_config = getattr(self.model_configs, architecture_name)

        # Create the model based on architecture type
        print(self.data_configs.image_size, self.data_configs.num_classes)
        model = model_class(model_config, self.data_configs.image_size, self.data_configs.num_classes)

        # Compile the model if requested
        if compile_model:
            model = self.compile_model(model)

        return model

    def compile_model(self, model: keras.Model) -> keras.Model:
        """
        Compile the model with the configured specifications.

        Args:
            model: The model to compile

        Returns:
            Compiled model
        """
        # Create optimizer based on configuration
        optimizer = self._create_optimizer()

        # Define metrics
        metrics = self._define_metrics()

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=self.compile_config.loss,
            metrics=metrics,
            steps_per_execution=16  # Performance optimization
        )

        # Clean up optimizer reference to prevent memory leaks
        del optimizer
        gc.collect()

        return model

    def _create_optimizer(self) -> keras.optimizers.Optimizer:
        """
        Create optimizer based on configuration.

        Returns:
            Configured optimizer

        Raises:
            ValueError: If optimizer type is not supported
        """
        optimizer_name = self.compile_config.optimizer.lower()
        learning_rate = self.compile_config.lr

        if optimizer_name == "adam":
            return keras.optimizers.Adam(learning_rate)
        elif optimizer_name == "sgd":
            return keras.optimizers.SGD(learning_rate)
        elif optimizer_name == "rmsprop":
            return keras.optimizers.RMSprop(learning_rate)
        elif optimizer_name == "adamw":
            return keras.optimizers.AdamW(learning_rate)
        else:
            supported = ["adam", "sgd", "rmsprop", "adamw"]
            raise ValueError(f"Unsupported optimizer '{optimizer_name}'. "
                             f"Supported: {supported}")

    def _define_metrics(self) -> List[Any]:
        """
        Define metrics based on configuration.

        Returns:
            List of configured metrics
        """
        metrics = []

        for metric_name in self.compile_config.metrics:
            metric_name_lower = metric_name.lower()

            if metric_name_lower == 'accuracy':
                metrics.append('accuracy')
            elif metric_name_lower == 'precision':
                metrics.append(keras.metrics.Precision(name='precision'))
            elif metric_name_lower == 'recall':
                metrics.append(keras.metrics.Recall(name='recall'))
            elif metric_name_lower == 'f1-score':
                metrics.append(keras.metrics.F1Score(name='f1-score'))
            elif metric_name_lower == 'top-3':
                metrics.append(keras.metrics.TopKCategoricalAccuracy(k=3, name='top-3'))
            elif metric_name_lower == 'top-5':
                metrics.append(keras.metrics.TopKCategoricalAccuracy(k=5, name='top-5'))
            elif metric_name_lower == 'auc':
                metrics.append(keras.metrics.AUC(name='auc'))
            else:
                print(f"Warning: Unknown metric '{metric_name}' ignored")

        return metrics
