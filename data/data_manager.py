import tensorflow as tf
from typing import Tuple, Optional
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from data_loader import DataLoader


class DataManager:
    """
    Manages all data loading and preparation.
    Handles both real and synthetic datasets and different training configurations.
    """

    def __init__(self, data_config, batch_size):
        """
        Initialize DataManager with data configuration.

        Args:
            data_config: Configuration object containing data paths and parameters
            batch_size: Batch size as defined in the config yaml
        """
        self.data_config = data_config
        self.batch_size = batch_size

        # Initialize data loaders
        self.real_data_loader = DataLoader(
            data_config.data_path_real,
            data_config.image_size
        )
        self.synthetic_data_loader = DataLoader(
            data_config.data_path_synth,
            data_config.image_size
        )

        self.load_datasets()
        self.AUTOTUNE = tf.data.AUTOTUNE

    def load_datasets(self):
        """Load and split all datasets."""

        self.real_data_loader.load_splits()
        self.synthetic_data_loader.load_splits()

    def get_validation_data(self) -> tf.data.Dataset:
        """
        Get prepared validation dataset.
        """

        val_data = self.real_data_loader.get_validation_split().dataset

        return self._prepare_dataset(val_data)

    def get_test_data(self) -> tf.data.Dataset:
        """
        Get prepared test dataset.

        """
        test_data = self.real_data_loader.get_test_split().dataset

        return self._prepare_dataset(test_data)

    def get_training_data(self, setting: str, seed: int,
                          proportion: int) -> DatasetV2 | tuple[DatasetV2, DatasetV2]:
        """
        Get training data based on the specified setting.

        Args:
            setting: Training setting ('simple_mixed' or 'fine-tuned')
            seed: For reproducible shuffling
            proportion: Proportion for mixing datasets (0-10 for simple_mixed, 1-9 for fine-tuned)

        Returns:
            For simple_mixed: train_data
            For fine-tuned: (pretrain_data, fine_tune_data)
        """

        if setting == "simple_mixed":
            return self._get_simple_mixed_data(seed, proportion)
        elif setting == "fine-tuned":
            return self._get_fine_tuned_data(seed, proportion)
        else:
            raise ValueError(f"Unknown training setting: {setting}")

    def _get_simple_mixed_data(self, seed: int,
                               proportion: int) -> tf.data.Dataset:
        """
        Prepare data for simple mixed training.

        Args:
            seed: Random seed for shuffling
            proportion: Proportion of synthetic data (0-10)

        Returns:
            train_data
        """
        if proportion < 0 or proportion > 10:
            raise ValueError("Proportion must be between 0 and 10 for simple_mixed setting")

        # Combine real and synthetic datasets based on proportion
        train_data_splits_list = (self.real_data_loader.get_train_splits()[proportion:] +
                                  self.synthetic_data_loader.get_train_splits()[:proportion])

        # Create hybrid dataset by interleaving
        train_data = self._combine_datasets([split.dataset for split in train_data_splits_list])

        # Prepare the training data with shuffling and batching
        buffer_size = sum(split.size for split in train_data_splits_list)
        train_data = self._prepare_dataset(train_data, buffer_size, seed)

        return train_data

    def _get_fine_tuned_data(self, seed: int,
                             proportion: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare data for fine-tuned training (pretraining + fine-tuning).

        Args:
            seed: Random seed for shuffling
            proportion: Number of synthetic datasets to use for pretraining (1-9)

        Returns:
            Tuple of (pretrain_data, fine_tune_data)
        """
        if proportion < 1 or proportion > 9:
            raise ValueError("Proportion must be between 1 and 9 for fine-tuned setting")

        # Create subsets based on proportion
        synth_train_data_splits = self.synthetic_data_loader.get_train_splits()[:proportion]
        real_train_data_splits = self.real_data_loader.get_train_splits()[proportion:]

        # Prepare synthetic data for pretraining
        synth_data_subset = self._combine_datasets([split.dataset for split in synth_train_data_splits])
        synth_buffer_size = sum(split.size for split in synth_train_data_splits)
        pretrain_data = self._prepare_dataset(synth_data_subset, synth_buffer_size, seed)

        # Prepare real data for fine-tuning
        real_data_subset = self._combine_datasets([split.dataset for split in real_train_data_splits])
        real_buffer_size = sum(split.size for split in real_train_data_splits)
        fine_tune_data = self._prepare_dataset(real_data_subset, real_buffer_size, seed)

        return pretrain_data, fine_tune_data

    def _prepare_dataset(self, dataset: tf.data.Dataset, shuffle_buffer: Optional[int] = None,
                         seed: Optional[int] = None) -> tf.data.Dataset:
        """Apply common dataset preparation steps."""
        dataset = dataset.cache()

        if shuffle_buffer:
            dataset = dataset.shuffle(shuffle_buffer, seed=seed)

        dataset = (dataset
                   .batch(self.batch_size, num_parallel_calls=self.AUTOTUNE)
                   # .apply(tf.data.experimental.copy_to_device('/GPU:0'))
                   .prefetch(self.AUTOTUNE))

        return dataset

    @staticmethod
    def _combine_datasets(dataset_list):
        """
        Combine multiple datasets into one, no shuffling.

        Args:
            dataset_list: List of tf.data.Dataset objects

        Returns:
            Combined dataset
        """
        if len(dataset_list) == 1:
            return dataset_list[0]
        else:
            combined = dataset_list[0]
            for ds in dataset_list[1:]:
                combined = combined.concatenate(ds)
            return combined
