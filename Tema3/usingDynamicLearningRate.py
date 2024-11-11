import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))

def download_mnist(is_train: bool):
    dataset = MNIST(
        root='./data',
        transform=transforms.Lambda(lambda x: np.array(x).flatten() / 255.0),  # Normalize pixel values
        download=True,
        train=is_train
    )
    mnist_data = np.array([image for image, _ in dataset])
    mnist_labels = np.array([label for _, label in dataset])
    return mnist_data, mnist_labels

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

num_classes = 10
train_Y_encoded = one_hot_encode(train_Y, num_classes)
test_Y_encoded = one_hot_encode(test_Y, num_classes)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)

def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1  
    a1 = relu(z1)           
    z2 = np.dot(a1, W2) + b2 
    return softmax(z2), a1  


def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
    return cross_entropy_loss


def train(X, Y, epochs=100, batch_size=50, initial_learning_rate=0.01, lr_decay_factor=0.5, patience=5, min_delta=0.001):
    m, n = X.shape 
    hidden_units = 100 
    W1 = np.random.randn(n, hidden_units) * 0.01 
    b1 = np.zeros(hidden_units)
    W2 = np.random.randn(hidden_units, num_classes) * 0.01
    b2 = np.zeros(num_classes)
    learning_rate = initial_learning_rate
    
    best_loss = float('inf')
    epochs_no_improve = 0 

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size]
            Y_batch = Y[i:i + batch_size]

            y_pred, a1 = forward_propagation(X_batch, W1, b1, W2, b2)
            batch_loss = compute_loss(y_pred, Y_batch)
            epoch_loss += batch_loss

            error = y_pred - Y_batch
            W2_grad = np.dot(a1.T, error) / batch_size
            b2_grad = np.sum(error, axis=0) / batch_size

            hidden_error = np.dot(error, W2.T) * (a1 > 0)
            W1_grad = np.dot(X_batch.T, hidden_error) / batch_size
            b1_grad = np.sum(hidden_error, axis=0) / batch_size

            W1 -= learning_rate * W1_grad
            b1 -= learning_rate * b1_grad
            W2 -= learning_rate * W2_grad
            b2 -= learning_rate * b2_grad

        epoch_loss /= (m // batch_size)
        
        if best_loss - epoch_loss > min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            learning_rate *= lr_decay_factor
            print(f'Learning rate decayed to: {learning_rate}')
            epochs_no_improve = 0

        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    y_pred, _ = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(y_pred, axis=1)

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

W1, b1, W2, b2 = train(train_X, train_Y_encoded, epochs=70, initial_learning_rate=0.1, batch_size=50, lr_decay_factor=0.5, patience=5, min_delta=0.001)

train_predictions = predict(train_X, W1, b1, W2, b2)
test_predictions = predict(test_X, W1, b1, W2, b2)

train_accuracy = compute_accuracy(train_predictions, train_Y)
test_accuracy = compute_accuracy(test_predictions, test_Y)

print(f'Train accuracy: {train_accuracy * 100:.2f}%')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

def plot_random_test_image(test_X, test_Y):
    index = np.random.randint(0, len(test_X))
    image = test_X[index].reshape(28, 28)
    true_label = test_Y[index]
    predicted_label = predict(test_X[index:index + 1], W1, b1, W2, b2)[0]
    
    plt.imshow(image, cmap='gray')
    plt.title(f'True Label: {true_label}, Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()

# for i in range(10):
#     plot_random_test_image(test_X, test_Y)

#best accuracy:100% train , 98.13% test