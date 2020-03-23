from dnet import datasets
from dnet.archs import VGG16
from dnet.layers import FC

(x_train, y_train), (x_val, y_val) = datasets.tiny_mnist(flatten=False, one_hot_encoding=True)

model = VGG16()()
model.add(FC(units=10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", lr=1e-03, bs=512)
model.fit(inputs=x_train, targets=y_train, epochs=10, validation_data=(x_val, y_val))

model.plot_losses()
model.plot_accuracy()
