from unittest import TestCase

from jax.interpreters.xla import _DeviceArray

from kerax.losses import BCELoss, CCELoss
from kerax.utils import jnp, random


class TestLossFunctions(TestCase):

    def setUp(self) -> None:
        self.keys = random.split(random.PRNGKey(42), 4)

    def testBCELoss(self):
        k1, k2 = self.keys[0], self.keys[1]
        binary_predictions = random.uniform(key=k1, shape=(100, 1))
        binary_labels = random.permutation(key=k2,
                                           x=jnp.concatenate((jnp.zeros(shape=(50,)), jnp.ones(shape=(50,)))))
        loss = BCELoss(predictions=binary_predictions, targets=binary_labels)
        self.assertIsInstance(loss, _DeviceArray)

    def testCCELoss(self):
        k3, k4 = self.keys[2], self.keys[3]
        softmax_predictions = random.randint(key=k3, shape=(100, 1), minval=0, maxval=9)
        softmax_labels = random.randint(key=k4, shape=(100, 1), minval=0, maxval=9)
        loss = CCELoss(predictions=softmax_predictions, targets=softmax_labels)
        self.assertIsInstance(loss, _DeviceArray)

    def tearDown(self) -> None:
        del self.keys
