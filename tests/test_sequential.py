from unittest import TestCase

from kerax.datasets import binary_tiny_mnist
from kerax.layers import Dense, Sigmoid
from kerax.losses import BCELoss
from kerax.metrics import binary_accuracy
from kerax.models import Sequential
from kerax.optimizers import SGD
from kerax.utils import Interpreter, device_put


class TestSequential(TestCase):

    def setUp(self) -> None:
        self.data = binary_tiny_mnist.load_dataset(batch_size=200)
        self.binary_model = Sequential(layers=[Dense(1), Sigmoid])

    def tearDown(self) -> None:
        del self.data
        del self.binary_model

    def test_attrs_after_init(self) -> None:
        self.assertNotEqual(self.binary_model._layers, None)
        self.assertEqual(self.binary_model._epochs, 1)
        self.assertEqual(self.binary_model._trained_params, [])
        self.assertEqual(self.binary_model._loss_fn, None)
        self.assertEqual(self.binary_model._optimizer, None)
        self.assertEqual(", ".join(self.binary_model._metrics.keys()), "loss, loss_per_epoch")
        self.assertEqual(self.binary_model._metrics_fn, [])
        self.assertEqual(self.binary_model._seed, 0)

    def test_add_layers(self) -> None:
        new_binary_model = Sequential([Dense(100)]) + self.binary_model
        self.assertEqual(len(new_binary_model._layers), 3)
        another_binary_model = Sequential([Dense(100)])
        another_binary_model.add(self.binary_model)
        self.assertEqual(len(another_binary_model._layers), 3)
        one_more_binary_model = Sequential([Dense([100])])
        one_more_binary_model.add([Dense(10), Sigmoid])
        self.assertEqual(len(one_more_binary_model._layers), 3)
        self.assertEqual(one_more_binary_model.add(1), None)

    def test_compile(self) -> None:
        loss_fn, opt_fn, metrics_fn = BCELoss, SGD(step_size=0.001), [binary_accuracy]
        self.binary_model.compile(loss=loss_fn, optimizer=opt_fn, metrics=metrics_fn)
        self.assertEqual(self.binary_model._loss_fn, loss_fn)
        self.assertEqual(self.binary_model._optimizer, opt_fn)
        self.assertEqual(self.binary_model._metrics_fn, metrics_fn)
        self.assertEqual(", ".join(self.binary_model._metrics.keys()), "loss, loss_per_epoch, binary_accuracy")

    def test_fit(self) -> None:
        self.binary_model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01), metrics=[binary_accuracy])
        self.binary_model.fit(data=self.data, epochs=1)
        self.assertNotEqual(self.binary_model._trained_params, None)
        self.assertEqual(len(self.binary_model._metrics.keys()), 3)

    def test_predict(self) -> None:
        self.binary_model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01), metrics=[binary_accuracy])
        self.binary_model.fit(data=self.data, epochs=1)
        inputs = device_put(next(self.data.train_data)[0])
        self.assertEqual(inputs.shape, (200, 784))
        outputs = self.binary_model.predict(inputs)
        self.assertEqual(outputs.shape, (200, 1))

    def test_save_and_load_and_train(self) -> None:
        from pathlib import Path
        self.binary_model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01), metrics=[binary_accuracy])
        self.binary_model.fit(data=self.data, epochs=1)
        self.binary_model.save("dummy_binary_model")
        path = Path(__file__).parent / "dummy_binary_model.msgpack"
        self.assertTrue(path.exists())
        loaded_model = Sequential()
        loaded_model.load("dummy_binary_model")
        self.assertEqual(len(loaded_model._layers), 2)
        self.assertNotEqual(loaded_model._optimizer, None)
        self.assertEqual(len(loaded_model._trained_params), 2)
        self.assertEqual(len(loaded_model._metrics_fn), 1)
        self.assertNotEqual(loaded_model._loss_fn, None)
        model = Sequential()
        model.load("dummy_binary_model")
        model.fit(data=self.data, epochs=1)

    def test_get_interpreter(self) -> None:
        self.binary_model.compile(loss=BCELoss, optimizer=SGD(step_size=0.01), metrics=[binary_accuracy])
        self.binary_model.fit(data=self.data, epochs=1)
        interp = self.binary_model.get_interpretation()
        self.assertIsInstance(interp, Interpreter)
        self.assertEqual(interp._config.get("epochs"), 1)
        self.assertEqual(len(interp._config.get("metrics").keys()), 3)
