from kerax.datasets import mnist, binary_tiny_mnist
from unittest import TestCase


class TestDatasets(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_binary_mnist(self) -> None:
        data = binary_tiny_mnist.load_dataset(batch_size=200)
        self.assertEqual(data.batch_size, 200)
        self.assertEqual(data.num_train_batches, 17)
        self.assertEqual(data.num_val_batches, 5)
        inputs, targets = next(data.train_data)
        self.assertEqual(inputs.shape, (200, 784))
        self.assertEqual(targets.shape, (200, 1))
        inputs, targets = next(data.val_data)
        self.assertEqual(inputs.shape, (200, 784))
        self.assertEqual(targets.shape, (200, 1))
        self.assertEqual(data.input_shape, (-1, 784))

    def test_mnist(self) -> None:
        data = mnist.load_dataset(batch_size=1000)
        self.assertEqual(data.batch_size, 1000)
        self.assertEqual(data.num_train_batches, 60)
        self.assertEqual(data.num_val_batches, 10)
        inputs, targets = next(data.train_data)
        self.assertEqual(inputs.shape, (1000, 28, 28, 1))
        self.assertEqual(targets.shape, (1000, 10))
        inputs, targets = next(data.val_data)
        self.assertEqual(inputs.shape, (1000, 28, 28, 1))
        self.assertEqual(targets.shape, (1000, 10))
        self.assertEqual(data.input_shape, (-1, 28, 28, 1))
