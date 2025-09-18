import tensorflow as tf
import keras
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import field, dataclass


@dataclass(slots=True)
class DatasetSplit:
    """Metadata and Dataset for a split."""
    name: str
    file_paths: Dict[str, List[str]]  # class_name -> list of file paths
    size: int
    class_distribution: Dict[str, int]
    dataset: Optional[tf.data.Dataset] = field(
        init=False,
        default=None,
        repr=False,
        compare=False
    )


class ImagePreprocessor:
    """Handles image loading and preprocessing."""

    def __init__(self, img_size: Tuple[int, int, int]):
        self.img_size = img_size

    def load_and_preprocess_image(self, path: tf.Tensor) -> tf.Tensor:
        """Load and preprocess a single image."""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.img_size[0], self.img_size[1]])
        image = image / 255.0
        image = tf.image.per_image_standardization(image)

        return image


class DatasetBuilder:
    """Builds TensorFlow datasets from file paths."""

    def __init__(self, preprocessor: ImagePreprocessor):
        self.preprocessor = preprocessor

    def build_dataset(self, split: DatasetSplit) -> tf.data.Dataset:
        """
        Build a TensorFlow dataset from a DatasetSplit.

        Args:
            split: DatasetSplit containing file paths and metadata

        Returns:
            TensorFlow dataset
        """
        # Create label mapping
        class_names = sorted(split.file_paths.keys())
        label_to_int = {name: i for i, name in enumerate(class_names)}

        # Collect all file paths and labels
        all_paths = []
        all_labels = []

        for class_name, file_paths in split.file_paths.items():
            all_paths.extend(file_paths)
            all_labels.extend([label_to_int[class_name]] * len(file_paths))

        # Convert labels to one-hot
        all_labels = keras.utils.to_categorical(all_labels, num_classes=len(class_names))

        # Create TensorFlow dataset
        path_ds = tf.data.Dataset.from_tensor_slices(all_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(all_labels)

        # Apply preprocessing
        image_ds = path_ds.map(
            self.preprocessor.load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return tf.data.Dataset.zip((image_ds, label_ds))


class DataLoader:
    """
    Data loader maintains perfect class balance and deterministic splits.
    """

    def __init__(self, data_path: str, img_size: Tuple[int, int, int]):
        self.data_path = Path(data_path)
        self.img_size = img_size

        # Initialize components
        self.preprocessor = ImagePreprocessor(img_size)
        self.dataset_builder = DatasetBuilder(self.preprocessor)

        # Dataset splits (validation: 20%, test: 20%, train: 10 splits of 6% each)
        self.split_ratios = [0.20, 0.20] + [0.06] * 10
        self.dataset_splits: Optional[List[DatasetSplit]] = None

    def _scan_directory(self) -> Dict[str, List[str]]:
        """
        Scan directory and organize files by class.

        Returns:
            a Dict where the keys are the classes (as strings) and
            the values are lists that contain the paths to the images (as strings)
        """
        class_files = {}

        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_files[class_name] = []

                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        class_files[class_name].append(str(file_path))

        return class_files

    def _create_balanced_splits(self, class_files: Dict[str, List[str]],
                                split_ratios: List[float]) -> List[DatasetSplit]:
        """
        Create perfectly balanced splits from class files.

        Args:
            class_files: Dictionary mapping class names to file paths
            split_ratios: List of ratios for each split (should sum to 1.0)

        Returns:
            List of DatasetSplit objects
        """
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        num_splits = len(split_ratios)
        splits = [DatasetSplit(f"split_{i}", {}, 0, {}) for i in range(num_splits)]

        for class_name, files in class_files.items():
            # Shuffle files deterministically
            files_copy = files.copy()
            random.shuffle(files_copy)

            # Calculate number of files per split
            total_files = len(files_copy)
            files_per_split = [int(total_files * ratio) for ratio in split_ratios]

            # Adjust for rounding errors
            diff = total_files - sum(files_per_split)
            for i in range(diff):
                files_per_split[i % num_splits] += 1

            # Distribute files to splits
            start_idx = 0
            for i, count in enumerate(files_per_split):
                end_idx = start_idx + count
                splits[i].file_paths[class_name] = files_copy[start_idx:end_idx]
                splits[i].class_distribution[class_name] = count
                splits[i].dataset = self.dataset_builder.build_dataset(splits[i])
                start_idx = end_idx

        # Calculate total sizes
        for split in splits:
            split.size = sum(split.class_distribution.values())

        return splits

    def load_splits(self) -> List[DatasetSplit]:
        """Create dataset splits (called once per data loader instance)."""
        if self.dataset_splits is not None:
            return self.dataset_splits

        # Create splits from directory scan
        class_files = self._scan_directory()
        self.dataset_splits = self._create_balanced_splits(class_files, self.split_ratios)

        return self.dataset_splits

    def get_validation_split(self) -> DatasetSplit:
        """Get validation split (index 0)."""
        splits = self.load_splits()
        return splits[0]

    def get_test_split(self) -> DatasetSplit:
        """Get test split (index 1)."""
        splits = self.load_splits()
        return splits[1]

    def get_train_splits(self) -> List[DatasetSplit]:
        """Get all training splits (indices 2-11)."""
        splits = self.load_splits()
        return splits[2:]