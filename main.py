import torch
import torchvision
import torchvision.transforms as transforms
from CNN1 import CNN1
from Trainer import Trainer
from Tester import Tester

if __name__ == '__main__':

    # Normalize dataset
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    # Load training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    # Load test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create and train the network
    trainer = Trainer(CNN1, trainloader)
    trainer.train()
    save_location = trainer.get_save_location()

    # Test the network on test dataset
    tester = Tester(CNN1, testloader)
    tester.load_network_from_file(save_location)
    tester.test_accuracy()
    tester.test_class_accuracy(classes)

    print("Finished")