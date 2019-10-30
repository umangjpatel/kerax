import pandas as pd

from dnet import *

# Loading the training dataset
trainer = pd.read_csv("datasets/mnist/mnist_train_small.csv", header=None)
train_data = trainer.loc[(trainer[0] == 1) | (trainer[0] == 0)]
train_features, train_labels = train_data.iloc[:, 1:].values, train_data[1].values
print("Training data -> features shape : {}, labels shape : {}".format(train_features.shape, train_labels.shape))

# Loading the training dataset
validator = pd.read_csv("datasets/mnist/mnist_test.csv", header=None)
val_data = validator.loc[(validator[0] == 1) | (validator[0] == 0)]
val_features, val_labels = val_data.iloc[:, 1:].values, val_data[1].values
print("Validation data -> features shape : {}, labels shape : {}".format(val_features.shape, val_labels.shape))

# Create model object
model = DNet()

# Define the model architecture
model.add(FC(units=500, activation='relu', keep_prob=0.5))
model.add(FC(units=50, activation='relu', keep_prob=0.5))
model.add(FC(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', epochs=20, lr=3e-2)

# Train the model
model.fit(train_features, train_labels)

# Plot the training loss curve
model.plot_losses()

# Evaluate the model on validation data
val_acc_score = model.evaluate(val_features, val_labels, threshold=1.0)
print("Validation accuracy : {0:.6}".format(val_acc_score))

# Make predictions on unseen data
# model.predict(test_features)
