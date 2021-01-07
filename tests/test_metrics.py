from unittest import TestCase

from jax.interpreters.xla import _DeviceArray
from kerax.utils import jnp, random
from kerax.metrics import binary_accuracy, accuracy


class TestMetricsFunctions(TestCase):

    def setUp(self) -> None:
        self.keys = random.split(random.PRNGKey(42), 4)

    def tearDown(self) -> None:
        del self.keys

    def test_binary_accuracy(self) -> None:
        k1, k2 = self.keys[0], self.keys[1]
        predictions = random.uniform(key=k1, shape=(100, 1), minval=0, maxval=1)
        labels = random.randint(key=k2, shape=(100, 1), minval=0, maxval=1)
        acc = binary_accuracy(predictions=predictions, targets=labels, acc_thresh=0.5)
        self.assertIsInstance(acc, _DeviceArray)

    def test_accuracy(self) -> None:
        k1, k2 = self.keys[2], self.keys[3]
        predictions = random.randint(key=k1, shape=(100, 1), minval=0, maxval=9)
        labels = random.randint(key=k2, shape=(100, 1), minval=0, maxval=9)
        acc = accuracy(predictions=predictions, targets=labels)
        self.assertIsInstance(acc, _DeviceArray)

