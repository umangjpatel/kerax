from unittest import TestCase

from kerax.layers import Dense, Relu, Sigmoid, LogSoftmax, Flatten, Dropout
from kerax.utils import stax, random, jnp


class TestLayers(TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @staticmethod
    def get_dummy_network(layers, inputs):
        init_params, forward_pass = stax.serial(*layers)
        input_shape = tuple([-1] + list(inputs.shape)[1:])
        _, params = init_params(random.PRNGKey(10), input_shape)
        return params, forward_pass

    @staticmethod
    def get_dummy_inputs(input_shape):
        return random.normal(key=random.PRNGKey(42), shape=input_shape)

    def test_dense_layer(self):
        inputs = self.get_dummy_inputs(input_shape=(200, 784))
        params, forward_pass = self.get_dummy_network([Dense(10)], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(outputs.shape, (200, 10))

    def test_relu_layer(self):
        inputs = self.get_dummy_inputs(input_shape=(200, 784))
        params, forward_pass = self.get_dummy_network([Relu], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_sigmoid_layer(self):
        inputs = self.get_dummy_inputs(input_shape=(200, 784))
        params, forward_pass = self.get_dummy_network([Sigmoid], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_flatten_layer_already_flat(self):
        inputs = self.get_dummy_inputs(input_shape=(200, 784))
        params, forward_pass = self.get_dummy_network([Flatten], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(inputs.shape, outputs.shape)

    def test_flatten_layer_not_already_flat(self):
        inputs = self.get_dummy_inputs(input_shape=(200, 28, 28, 1))
        params, forward_pass = self.get_dummy_network([Flatten], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(outputs.shape, (200, 784))

    def test_log_softmax_layer(self):
        inputs = self.get_dummy_inputs(input_shape=(200, 784))
        params, forward_pass = self.get_dummy_network([LogSoftmax], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(outputs.shape, (200, 784))

    def test_dropout_layer_mode_train(self):
        inputs = self.get_dummy_inputs(input_shape=(1, 5))
        params, forward_pass = self.get_dummy_network([Dropout(rate=0.0)], inputs)
        outputs = forward_pass(params, inputs)
        self.assertEqual(len(outputs == 0.0), len(inputs))

    def test_dropout_layer_mode_predict(self):
        inputs = self.get_dummy_inputs(input_shape=(1, 5))
        params, forward_pass = self.get_dummy_network([Dropout(rate=0.0)], inputs)
        outputs = forward_pass(params, inputs, mode="predict")
        self.assertEqual(len(outputs != 0.0), len(inputs))

