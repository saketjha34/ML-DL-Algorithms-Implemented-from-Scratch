import os  ##-> for directory and num_workers
from torch.utils.data import DataLoader ##-> For MNIST Dataset
from torchvision import datasets, transforms ##-> For MNIST Dataset
import matplotlib.pyplot as plt ##-> for plotting loss curves
import numpy as np  ## -> Math
from tqdm import tqdm ## -> For Progressbar
from sklearn.metrics import accuracy_score ##-> accuracy
import torch

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        ]),

    'test': transforms.Compose([
        transforms.ToTensor(),
        ]),
}

train_data = datasets.MNIST(
    train = True,
    root = 'data/',
    download = True,
    transform = data_transforms['train'],
    target_transform=None,
)

test_data = datasets.MNIST(
    root="data/",
    train=False,
    download=True,
    transform=data_transforms['test'],
    target_transform=None,
)


NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count()

train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = True,
)

test_dataloader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    num_workers = NUM_WORKERS,
    shuffle = False
)

class SimpleMLP:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights, self.biases = self._initialize_weights()

    def _initialize_weights(self):
        np.random.seed(0)
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01)
            biases.append(np.zeros((1, self.layer_sizes[i + 1])))
        return weights, biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        activations = [X]
        input = X
        for w, b in zip(self.weights, self.biases):
            output = self.sigmoid(np.dot(input, w) + b)
            activations.append(output)
            input = output
        return activations

    def backward(self, X, y, activations):
        deltas = [None] * len(self.weights)
        error = y - activations[-1]
        deltas[-1] = error * self.sigmoid_derivative(activations[-1])
        
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] = deltas[i + 1].dot(self.weights[i + 1].T) * self.sigmoid_derivative(activations[i + 1])

        for i in range(len(self.weights)):
            self.weights[i] += activations[i].T.dot(deltas[i]) * self.learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def compute_accuracy(self, predictions, labels):
        return np.mean(predictions == labels) * 100

    def train(self, train_loader, test_loader, epochs=10):
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            train_loss, test_loss = 0, 0
            correct_train, total_train = 0, 0
            correct_test, total_test = 0, 0

            # Training
            for X, y in train_loader:
                X = X.view(-1, 28 * 28).numpy()
                y_onehot = np.eye(10)[y.numpy()]

                activations = self.forward(X)
                self.backward(X, y_onehot, activations)

                loss = self.compute_loss(y_onehot, activations[-1])
                train_loss += loss

                predictions = np.argmax(activations[-1], axis=1)
                correct_train += (predictions == y.numpy()).sum()
                total_train += y.size(0)

            # Testing
            for X, y in test_loader:
                X = X.view(-1, 28 * 28).numpy()
                y_onehot = np.eye(10)[y.numpy()]

                activations = self.forward(X)
                loss = self.compute_loss(y_onehot, activations[-1])
                test_loss += loss

                predictions = np.argmax(activations[-1], axis=1)
                correct_test += (predictions == y.numpy()).sum()
                total_test += y.size(0)

            # Average losses and accuracies
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            train_accuracy = correct_train / total_train * 100
            test_accuracy = correct_test / total_test * 100

            # Store losses and accuracies
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.7f}, Test Loss: {test_loss:.7f}, "
                  f"Train Acc: {train_accuracy:.7f}%, Test Acc: {test_accuracy:.7f}%")

        return train_losses, test_losses, train_accuracies, test_accuracies

    def plot_metrics(self, train_losses, test_losses, train_accuracies, test_accuracies):
        epochs = range(1, len(train_losses) + 1)

        # Plotting Loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train vs Test Loss')
        plt.legend()

        # Plotting Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, test_accuracies, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Train vs Test Accuracy')
        plt.legend()

        plt.show()

    def predict(self, data_loader):
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for X, y in data_loader:
                X = X.view(-1, 28 * 28).numpy()  # Flatten and convert to numpy for compatibility
                activations = self.forward(X)
                predictions = np.argmax(activations[-1], axis=1)
                
                all_predictions.extend(predictions)
                all_labels.extend(y.numpy())

        return np.array(all_predictions), np.array(all_labels)
    

layer_sizes = [784, 64, 10]  # Input layer, hidden layer with 64 neurons, output layer with 10 neurons
mlp = SimpleMLP(layer_sizes, learning_rate=0.01)
train_losses, test_losses, train_accuracies, test_accuracies = mlp.train(train_dataloader, test_dataloader, epochs=20)
    
train_preds , train_targets = mlp.predict(train_dataloader)
test_preds , test_targets = mlp.predict(test_dataloader)

print(f'Training Accuarcy : {accuracy_score(train_targets, train_preds)*100}%')
print(f'Test Accuarcy : {accuracy_score(test_targets, test_preds)*100}%')
    
mlp.plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)