import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import torchvision
from giza_actions.action import Action, action
from giza_actions.task import task
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Definition
input_size = 196  # 14x14
hidden_size = 10
num_classes = 10
num_epochs = 10
batch_size = 256
learning_rate = 0.001


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


# Task Definitions
def resize_images(images):
    return np.array([zoom(image[0], (0.5, 0.5)) for image in images])


@task(name="Prepare Datasets")
def prepare_datasets():
    print("Prepare dataset...")
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False)

    x_train = resize_images(train_dataset)
    x_test = resize_images(test_dataset)

    x_train = torch.tensor(x_train.reshape(-1, 14 * 14).astype("float32") / 255)
    y_train = torch.tensor([label for _, label in train_dataset], dtype=torch.long)

    x_test = torch.tensor(x_test.reshape(-1, 14 * 14).astype("float32") / 255)
    y_test = torch.tensor([label for _, label in test_dataset], dtype=torch.long)

    print("✅ Datasets prepared successfully")

    return x_train, y_train, x_test, y_test


@task(name="Create Loaders")
def create_data_loaders(x_train, y_train, x_test, y_test):
    print("Create loaders...")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    print("✅ Loaders created!")

    return train_loader, test_loader


@task(name="Train model")
def train_model(train_loader):
    print("Train model...")

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).reshape(-1, 14 * 14)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

    print("✅ Model trained successfully")
    return model


@task(name="Test model")
def test_model(model, test_loader):
    print("Test model...")
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device).reshape(-1, 14 * 14)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network on the 10000 test images: {acc} %")


@task(name="Convert To ONNX")
def convert_to_onnx(model, onnx_file_path):
    dummy_input = torch.randn(1, input_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
    )

    print(f"Model has been converted to ONNX and saved as {onnx_file_path}")


@action(name="Action: Convert To ONNX", log_prints=True)
def execution():
    x_train, y_train, x_test, y_test = prepare_datasets()
    train_loader, test_loader = create_data_loaders(x_train, y_train, x_test, y_test)
    model = train_model(train_loader)
    test_model(model, test_loader)

    # Convert to ONNX
    onnx_file_path = "mnist_model.onnx"
    convert_to_onnx(model, onnx_file_path)


if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-mnist-action")
    action_deploy.serve(name="pytorch-mnist-deployment")
