from unittest import TestCase

import numpy as np

from kerax.datasets import binary_tiny_mnist


class TestDataloader(TestCase):

    def setUp(self) -> None:
        self.data_loader = binary_tiny_mnist.load_dataset(batch_size=200)

    def tearDown(self) -> None:
        del self.data_loader

    def test_dataloader_attrs(self):
        self.assertEqual(self.data_loader.num_train_batches, 17)
        self.assertEqual(self.data_loader.num_val_batches, 5)
        self.assertEqual(self.data_loader.input_shape, (-1, 784))
        self.assertEqual(self.data_loader.batch_size, 200)

    def test_data_shapes(self):
        self.assertEqual(next(self.data_loader.train_data)[0].shape, (200, 784))
        self.assertEqual(next(self.data_loader.train_data)[1].shape, (200, 1))
        self.assertEqual(next(self.data_loader.val_data)[0].shape, (200, 784))
        self.assertEqual(next(self.data_loader.val_data)[1].shape, (200, 1))

    def test_data_items(self):
        train_item = next(self.data_loader.train_data)
        val_item = next(self.data_loader.train_data)
        train_inputs, train_labels = train_item
        val_inputs, val_labels = val_item
        self.assertIsInstance(train_item, tuple)
        self.assertIsInstance(val_item, tuple)
        self.assertIsInstance(train_inputs, np.ndarray)
        self.assertIsInstance(train_labels, np.ndarray)
        self.assertIsInstance(val_inputs, np.ndarray)
        self.assertIsInstance(val_labels, np.ndarray)
