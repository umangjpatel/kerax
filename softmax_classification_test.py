import os

import pandas as pd

from dnet import *

# Loading the paths to the data
data_path = os.path.join("datasets", "mnist")
train_data_path = os.path.join(data_path, 'mnist_train_small.csv')
test_data_path = os.path.join(data_path, "mnist_test.csv")

# Loading the training dataset
trainer = pd.read_csv(train_data_path, header=None)
train_features, train_labels = trainer.iloc[:, 1:].values, trainer[0].values.reshape(-1, 1)
print("Training data -> features shape : {}, labels shape : {}".format(train_features.shape, train_labels.shape))

# Loading the validation dataset
validator = pd.read_csv(test_data_path, header=None)
val_features, val_labels = validator.iloc[:, 1:].values, validator[0].values.reshape(-1, 1)
print("Validation data -> features shape : {}, labels shape : {}".format(val_features.shape, val_labels.shape))

# Create model object
model = DNet()

# Define the model architecture
model.add(FC(units=500, activation='relu'))
model.add(FC(units=50, activation='relu'))
model.add(FC(units=10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', epochs=20, lr=1e-2)

# Train the model
model.fit(train_features, train_labels)

# Plot the training loss curve
model.plot_losses()

# Evaluate the model on validation data
val_acc_score = model.evaluate(val_features, val_labels)
print("Validation accuracy : {0:.6}".format(val_acc_score))

# Make predictions on unseen data
# model.predict(test_features)
