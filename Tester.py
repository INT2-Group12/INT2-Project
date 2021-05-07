import torch

class Tester:

    def __init__(self, net, dataloader, batch_size=32):
        self._net = net()
        self._data_loader = dataloader
        self._batch_size = batch_size
        self._net_loaded = False

    def load_network_from_file(self, path):
        self._net.load_state_dict(torch.load(path))
        self._net_loaded = True

    def load_network(self, network):
        self._net = network
        self._net_loaded = True

    def test_accuracy(self):

        if not self._net_loaded:
            raise Exception("Network has not been loaded.")

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self._data_loader:
                images, labels = data
                outputs = self._net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total * 100
        print('Accuracy of the network on the 10000 test images: %.2f %%' % accuracy)

    def test_class_accuracy(self, classes):

        if not self._net_loaded:
            raise Exception("Network has not been loaded.")

        correct_classes = {classname: 0 for classname in classes}
        total_classes = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in self._data_loader:
                images, labels = data
                outputs = self._net(images)
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_classes[classes[label]] += 1
                    total_classes[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_classes.items():
            accuracy = 100 * float(correct_count) / total_classes[classname]
            print("Accuracy for class {:5s} is: {:.2f} %".format(classname, accuracy))
