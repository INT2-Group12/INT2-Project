import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np

class Trainer:

    def __init__(self, net, dataloader):
        self._net = net()
        self._data_loader = dataloader
        self._save_location = './cifar_net.pth'

    def _set_device(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net.to(device)
        return device

    def save_network(self):
        torch.save(self._net.state_dict(), self._save_location)

    def get_network(self):
        return self._net

    def get_save_location(self):
        return self._save_location

    def train(self):

        # Define Loss function and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adamax(self._net.parameters(), lr=0.001)

        # Use GPU if available
        device = self._set_device()

        start_time = time.perf_counter()
        train_loss_history = []
        train_acc_history = []

        # Train the network
        for epoch in range(10):

            running_loss = 0.0
            train_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(self._data_loader, 0):
                # data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # clear the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self._net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # calculate training accuracy and loss
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                train_loss += loss.item()

                # print loss and accuracy every 500 mini-batches
                running_loss += loss.item()
                if i % 500 == 499:
                    print('Epoch %d/10, %5d mini-batches, Loss: %.3f, Accuracy: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500, correct / total))
                    running_loss = 0.0

            train_loss_history.append(train_loss/len(self._data_loader))
            train_acc_history.append(correct/total)

        # print training time
        end_time = time.perf_counter()
        print(f'Finished training in {(end_time - start_time)/60:.2f} minutes.')

        # plot training accuracy and loss curve
        plt.plot(np.array(train_loss_history), 'b', label='Training Loss')
        plt.plot(np.array(train_acc_history), 'y', label='Training Accuracy')
        plt.legend()
        plt.show()

        self.save_network()