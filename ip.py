import numpy as np
import os

class BinaryDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def _load_binary_data(self, filename):
        """
        Loads binary data from file.
        - Images: stored as uint8, column-major (RGB channels stored separately).
        - Labels: stored as uint8, 1 per image.
        """
        path = os.path.join(self.dataset_path, filename)
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)

        return data

    def load_data(self, images_file, labels_file=None):
        """
        Loads images and labels (if available).
        - Images are reshaped to (96,96,3)
        - Labels are 1D array of integers (1-10)
        """
        images = self._load_binary_data(images_file)
        num_images = len(images) // (96 * 96 * 3)
        images = images.reshape(num_images, 3, 96, 96).transpose(0, 2, 3, 1)

        if labels_file:
            labels = self._load_binary_data(labels_file)
            return images, labels
        return images, None

    def load_unlabeled_data(self):
        """Loads the unlabeled dataset for inference."""
        return self.load_data("unlabeled.bin")[0]
