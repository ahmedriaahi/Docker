import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 3e-3
BATCH_SIZE = 64
EPOCHS = 8

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape data
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0
train_images = np.expand_dims(train_images, axis=-1)  # Shape: (28, 28, 1)
test_images = np.expand_dims(test_images, axis=-1)
train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)

# Define the CNN model
model = tf.Module()
initializer = tf.initializers.GlorotUniform()

# Define layers manually without Keras
model.conv1_weights = tf.Variable(initializer([5, 5, 1, 16]), trainable=True)
model.conv1_bias = tf.Variable(tf.zeros([16]), trainable=True)

model.conv2_weights = tf.Variable(initializer([3, 3, 16, 16]), trainable=True)
model.conv2_bias = tf.Variable(tf.zeros([16]), trainable=True)

model.fc1_weights = tf.Variable(initializer([400, 64]), trainable=True)
model.fc1_bias = tf.Variable(tf.zeros([64]), trainable=True)

model.fc2_weights = tf.Variable(initializer([64, 32]), trainable=True)
model.fc2_bias = tf.Variable(tf.zeros([32]), trainable=True)

model.fc3_weights = tf.Variable(initializer([32, 10]), trainable=True)
model.fc3_bias = tf.Variable(tf.zeros([10]), trainable=True)

def cnn_forward_pass(x):
    # Conv1 + ReLU + MaxPool
    x = tf.nn.conv2d(x, model.conv1_weights, strides=[1, 1, 1, 1], padding="VALID") + model.conv1_bias
    x = tf.nn.relu(x)
    x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    
    # Conv2 + ReLU + MaxPool
    x = tf.nn.conv2d(x, model.conv2_weights, strides=[1, 1, 1, 1], padding="VALID") + model.conv2_bias
    x = tf.nn.relu(x)
    x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    
    # Flatten
    x = tf.reshape(x, [-1, 400])
    
    # Dense layers
    x = tf.nn.relu(tf.matmul(x, model.fc1_weights) + model.fc1_bias)
    x = tf.nn.relu(tf.matmul(x, model.fc2_weights) + model.fc2_bias)
    logits = tf.matmul(x, model.fc3_weights) + model.fc3_bias
    return logits

# Loss and optimizer
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.nn.softmax_cross_entropy_with_logits

# Training and testing loop
def train_one_epoch():
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            logits = cnn_forward_pass(images)
            loss = tf.reduce_mean(loss_fn(labels, logits))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def evaluate():
    losses = []
    accuracies = []
    for images, labels in test_dataset:
        logits = cnn_forward_pass(images)
        loss = tf.reduce_mean(loss_fn(labels, logits))
        losses.append(loss.numpy())
        predictions = tf.argmax(logits, axis=1)
        labels = tf.argmax(labels, axis=1)
        accuracy = np.mean(predictions.numpy() == labels.numpy())
        accuracies.append(accuracy)
    return np.mean(losses), np.mean(accuracies)

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_one_epoch()
    test_loss, test_accuracy = evaluate()
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the model weights
np.savez("cnn_weights.npz", **{var.name: var.numpy() for var in model.trainable_variables})

