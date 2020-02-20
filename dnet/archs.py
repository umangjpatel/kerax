from dnet.layers import Conv2D, MaxPool2D, FC, Flatten
from dnet.models import Sequential


class Arch:
    pass


class LeNet5(Arch):

    def __init__(self) -> None:
        self.model: Sequential = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(5, 5), activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(FC(units=120, activation="relu"))
        self.model.add(FC(units=84, activation="relu"))

    def __call__(self) -> Sequential:
        return self.model
