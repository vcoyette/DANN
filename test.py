"""Functions to test."""
import torch


def testBaseLine(network, testloader):
    """Test the network on the test data loaer.

    Keyword Arguments:
        network -- the neural network to test
        device -- the device on which the network is
        testloader -- the dataloader for the dataset
    """
    device = next(network.parameters()).device

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100


def testDANN(network, testloader):
    """Test the network on the test data loaer.

    Keyword Arguments:
        network -- the neural network to test
        device -- the device on which the network is
        testloader -- the dataloader for the dataset
    """
    device = next(network.parameters()).device

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100
