# matchingnets.py
# This module contains the neural network model (inside the object MatchingNetworks).

import time

import numpy as np
from torch import Tensor
from torch import nn
import torch
from torchvision.models import vgg16, VGG16_Weights


class VGG16Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.transfer_vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        self.cnn_features = self.transfer_vgg16.features
        self.cnn_avgpool = self.transfer_vgg16.avgpool

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn_features(x)
        return self.cnn_avgpool(x)


class ClassifierNetwork(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.input = nn.Linear(in_features, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.output(x)


class MatchingNetworks:
    def __init__(self, train_loader, test_loader):
        self._cnn_network = VGG16Features()
        self._cnn_network.eval()
        self._classifier_network = ClassifierNetwork(
            in_features=self._cnn_network.cnn_avgpool.output_size[0]
        )
        self._lossfun = nn.MSELoss()
        self._optimizer = torch.optim.Adam(
            self._classifier_network.parameters(), lr=.01
        )
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._train_labels = torch.tensor([
            element[1] for element in self._train_loader
        ])
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def train(self, num_epochs=10):
        losses = torch.zeros(num_epochs)
        accuracy = []
        self._classifier_network.train()

        for epochi in range(num_epochs):
            print(f"Epoch {epochi + 1}")
            batch_loss = []
            for input, label in self._train_loader:
                input = input.to(self._device, dtype=torch.float32)
                label = label.to(self._device, dtype=torch.float32)
                # print()
                # print("input", input.dtype)
                # print("input", input.shape)
                # print("label", label.dtype)
                # print("label", label)

                cnn_outputs = self._cnn_network(input)
                # print("cnn_outputs", cnn_outputs.dtype)
                # print("cnn_outputs", cnn_outputs.shape)

                # forward pass and loss
                outputs = self._classifier_network(cnn_outputs)
                # print("outputs", outputs.dtype)
                # print("outputs", outputs.shape)
                loss = self._lossfun(outputs, label)

                # backprop
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                # loss from this batch
                batch_loss.append(loss.item())

            # and get average losses across the batches
            losses[epochi] = np.mean(batch_loss)

            # compute train accuracy
            input, label = next(iter(self._train_loader))
            with torch.no_grad():
                outputs = self._classifier_network(cnn_outputs)
            accuracy.append(
                100 * torch.mean(
                    (torch.abs(outputs - self._train_labels) < 1).float()
                )
            )

            # # compute test accuracy
            # X, y = next(iter(self._test_loader))
            # with torch.no_grad():
            #     y_hat = self._classifier_network(X)
            # test_accuracy.append(
            #     100 * torch.mean((torch.abs(y_hat - test_results) < 1).float())
            # )

        self._classifier_network.eval()
        return accuracy, losses

    def predict(self, loader):
        with torch.no_grad():
            y_hat = self._classifier_network(loader.dataset.tensors[0])
        return torch.round(y_hat.detach())
