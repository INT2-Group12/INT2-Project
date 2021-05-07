import torch

class Tester:
    """
    This is a class for testing a Convolutional Neural Network.

    Attributes:
        net (CNN1): A subclass of nn.Module.
        dataloader (DataLoader): Iterable over a dataset.
        batch_size (int): Size of the batches, 32 by default.
    """

    def __init__(self, net, dataloader, batch_size=32):
        """
        The constructor for Tester class.

        Args:
            net (CNN1): A subclass of nn.Module.
            dataloader (DataLoader): Iterable over a dataset.
            batch_size (int): Size of the batches, 32 by default.
        """
        self._net = net()
        self._data_loader = dataloader
        self._batch_size = batch_size
        self._net_loaded = False

    def load_network_from_file(self, path):
        """
        Load the model from the specified location.

        Args:
            path (str): Location of the model.
        """
        self._net.load_state_dict(torch.load(path))
        self._net_loaded = True

    def load_network(self, network):
        """
        Use the passed model.

        Args:
            network (CNN1): Model, an instance of class CNN1.
        """
        self._net = network
        self._net_loaded = True

    def test_accuracy(self):
        """
        Test the whole model and print the result.
        """
        if not self._net_loaded:
            raise Exception("Network has not been loaded.")

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self._data_loader:
                inputs, labels = data
                outputs = self._net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print('Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)

    def test_class_accuracy(self, classes):
        """
        Test each class separately and print the results.
        """
        if not self._net_loaded:
            raise Exception("Network has not been loaded.")

        correct_classes = {classname: 0 for classname in classes}
        total_classes = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in self._data_loader:
                inputs, labels = data
                outputs = self._net(inputs)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_classes[classes[label]] += 1
                    total_classes[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_classes.items():
            accuracy = 100 * float(correct_count) / total_classes[classname]
            print("Accuracy for class {:5s} is: {:.2f} %".format(classname, accuracy))