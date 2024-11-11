import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import time
np.random.seed(int(time.time()))
def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                     transform=transforms.Lambda(lambda x: np.array(x).flatten() / 255.0),  # Normalizare (impartim la 255 , valoaera maxima a unui pixel)
                     download=True,
                     train=is_train)
    mnist_data = np.array([image for image, _ in dataset])
    mnist_labels = np.array([label for _, label in dataset])
    return mnist_data, mnist_labels

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels] # pentru one_hot_encode formam matricea identitate pentru fiecare cifra si returnam linia pt acea clasa

num_classes = 10
train_Y_encoded = one_hot_encode(train_Y, num_classes)
test_Y_encoded = one_hot_encode(test_Y, num_classes)


def softmax(z):
    # Scadem maximul pentru a obtine doar valori intre 0 si 1, astfel incat pt 0 sa obitnem e^0=1, astfel ne protejam de overflow pt valori uriase
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    #returnam norma vectorului impartind fiecare element la suma tuturor elementelor pentru a obtine distributia probabilitatilor
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 

def forward_propagation(X, W, b):

    #pt forward prop. vom face dot product intre input (in cazul nostru un Battch de date) si weights si adaugam biasul fiecarui perceptron
    z = np.dot(X, W) + b
    #pe urma returnam outputl functiei softmax pentru acest set de date pentru a obtine distributia probabilitatilor
    return softmax(z)

def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m  # aici adaugam un numar foarte mic pentru a nu obtine niciodata log(0)


def train(X, Y, epochs=100, batch_size=50, learning_rate=0.01):
    m, n = X.shape # in cazul nostru m va fi nr de imagini iar n va fi numarul de pixeli ai unei imagini
    W = np.random.randn(n, num_classes) * 0.01  # Initializam Weight-urile pentru fiecare clasa cu numere random folosind deviateia standard 
    b = np.zeros(num_classes)  # initializam biasurile cu 0 

    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            #trimitem cate 100 de date pentru antrenament astfel incat sa reducem din complexitate
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]

            # Forward propagation
            y_pred = forward_propagation(X_batch, W, b)
            loss = compute_loss(y_pred, Y_batch)

            # Backward propagation
            error = y_pred - Y_batch  
            W_grad = np.dot(X_batch.T, error) / batch_size 
            b_grad = np.sum(error, axis=0) / batch_size  

            # Actualizam costurile si biasurile
            W -= learning_rate * W_grad
            b -= learning_rate * b_grad

        if epoch % 10 == 0:  # Afisam lossul la fiecare 10 epoci pt simplitae
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return W, b


def predict(X, W, b):
    y_pred = forward_propagation(X, W, b)
    return np.argmax(y_pred, axis=1)

def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


W, b = train(train_X, train_Y_encoded, epochs=100, learning_rate=0.1, batch_size=100)

train_predictions = predict(train_X, W, b)
test_predictions = predict(test_X, W, b)

train_accuracy = compute_accuracy(train_predictions, train_Y)
test_accuracy = compute_accuracy(test_predictions, test_Y)

print(f'Train accuracy: {train_accuracy * 100:.2f}%')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


def plot_random_test_image(test_X, test_Y):
    index = np.random.randint(0, len(test_X)) 
    image = test_X[index].reshape(28, 28)  
    true_label = test_Y[index]
    plt.imshow(image, cmap='gray')
    predicted_label = predict(test_X[index:index+1], W, b)[0]
    plt.title(f'True Label: {true_label}, Predicted: {predicted_label}')

    plt.axis('off')
    plt.show()

for i in range(10):
    plot_random_test_image(test_X,test_Y)
