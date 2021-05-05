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
        optimizer = optim.SGD(self._net.parameters(), lr=0.001, momentum=0.9)

        # Use GPU if available
        device = self._set_device()

        start_time = time.perf_counter()
        train_loss_history = []

        # Train the network
        for epoch in range(30):

            running_loss = 0.0
            train_loss = 0.0

            for i, data in enumerate(self._data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self._net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics and record history
                running_loss += loss.item()
                if i % 500 == 499:    # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0

                # calculate training accuracy and loss
                train_loss += loss.item()

            train_loss_history.append(train_loss/len(self._data_loader))

        end_time = time.perf_counter()
        print(f'Finished training in {(end_time - start_time)/60:.2f} minutes.')
        plt.plot(np.array(train_loss_history), 'r')
        plt.show()

        self.save_network()